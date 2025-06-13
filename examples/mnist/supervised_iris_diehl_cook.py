import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights

# 加载鸢尾花数据集
def load_iris_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

# 命令行参数设置
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--n_clamp", type=int, default=1)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=100)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=20)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--device_id", type=int, default=0)
parser.set_defaults(plot=False, gpu=True, train=True)

args = parser.parse_args()
seed = args.seed
n_neurons = args.n_neurons
batch_size = args.batch_size
n_clamp = args.n_clamp
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu
device_id = args.device_id

# 设置GPU使用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

# 加载鸢尾花数据
X_train, X_test, y_train, y_test = load_iris_data()
n_train = len(X_train)
n_test = len(X_test)

if not train:
    update_interval = n_test

n_classes = 3  # 鸢尾花有3个类别
n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity
per_class = int(n_neurons / n_classes)

# 构建Diehl & Cook 2015网络
# 注意：鸢尾花数据集有4个特征，而不是784个
network = DiehlAndCook2015(
    n_inpt=4,  # 鸢尾花有4个特征
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    dt=dt,
    nu=[1e-4, 1e-2],  # 调整学习率
    norm=1.0,  # 调整权重归一化因子
    theta_plus=theta_plus,
    inpt_shape=(1, 2, 2),  # 将4个特征重塑为2x2
)

# 将网络转移到GPU
if gpu:
    network.to("cuda")

# 兴奋层和抑制层的电压记录
exc_voltage_monitor = Monitor(network.layers["Ae"], ["v"], time=time, device=device)
inh_voltage_monitor = Monitor(network.layers["Ai"], ["v"], time=time, device=device)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")

# 编码器 - 将鸢尾花特征转换为脉冲
encoder = PoissonEncoder(time=time, dt=dt)

# 记录脉冲
spike_record = torch.zeros(update_interval, time, n_neurons, device=device)

# 神经元分配和脉冲比例
assignments = -torch.ones_like(torch.Tensor(n_neurons), device=device)
proportions = torch.zeros_like(torch.Tensor(n_neurons, n_classes), device=device)
rates = torch.zeros_like(torch.Tensor(n_neurons, n_classes), device=device)

# 准确率记录
accuracy = {"all": [], "proportion": []}

# 标签记录
labels = torch.empty(update_interval, device=device)

# 添加脉冲监视器
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

# 训练网络
print("Begin training.\n")

# 可视化变量初始化
inpt_axes = None
inpt_ims = None
spike_axes = None
spike_ims = None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes = None
voltage_ims = None

pbar = tqdm(total=n_train)
for i in range(n_train):
    # 获取当前样本
    x = X_train[i]
    label = y_train[i]
    
    # 将特征归一化到[0,1]范围，然后乘以强度
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8) * intensity
    
    # 编码为脉冲
    spike_train = encoder(torch.Tensor(x_norm))
    
    # 每个update_interval计算一次准确率
    if i % update_interval == 0 and i > 0:
        # 获取网络预测
        all_activity_pred = all_activity(spike_record, assignments, n_classes)
        proportion_pred = proportion_weighting(spike_record, assignments, proportions, n_classes)
        
        # 计算网络准确率
        accuracy["all"].append(
            100 * torch.sum(labels.long() == all_activity_pred).item() / update_interval
        )
        accuracy["proportion"].append(
            100 * torch.sum(labels.long() == proportion_pred).item() / update_interval
        )
        
        print(
            "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
            % (accuracy["all"][-1], np.mean(accuracy["all"]), np.max(accuracy["all"]))
        )
        print(
            "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f (best)\n"
            % (
                accuracy["proportion"][-1],
                np.mean(accuracy["proportion"]),
                np.max(accuracy["proportion"]),
            )
        )
        
        # 为兴奋层神经元分配标签
        assignments, proportions, rates = assign_labels(
            spike_record, labels, n_classes, rates
        )
    
    # 添加当前标签到标签列表
    labels[i % update_interval] = label
    
    # 运行网络
    choice = np.random.choice(int(n_neurons / n_classes), size=n_clamp, replace=False)
    clamp = {"Ae": per_class * torch.tensor(label, device=device).long() + torch.tensor(choice, device=device).long()}
    
    if gpu:
        inputs = {"X": spike_train.cuda().view(time, 1, 2, 2)}
    else:
        inputs = {"X": spike_train.view(time, 1, 2, 2)}
    
    network.run(inputs=inputs, time=time, clamp=clamp)
    
    # 获取电压记录
    exc_voltages = exc_voltage_monitor.get("v")
    inh_voltages = inh_voltage_monitor.get("v")
    
    # 添加到脉冲记录
    spike_record[i % update_interval] = spikes["Ae"].get("s").view(time, n_neurons)
    
    # 可选地绘制各种模拟信息
    if plot:
        # 将输入重塑为2x2以便可视化
        inpt = inputs["X"].view(time, 4).sum(0).view(2, 2)
        input_exc_weights = network.connections[("X", "Ae")].w
        square_weights = get_square_weights(
            input_exc_weights.view(4, n_neurons), n_sqrt, 2
        )
        square_assignments = get_square_assignments(assignments, n_sqrt)
        voltages = {"Ae": exc_voltages, "Ai": inh_voltages}
        
        inpt_axes, inpt_ims = plot_input(
            torch.zeros(2, 2), inpt, label=torch.tensor([label]), axes=inpt_axes, ims=inpt_ims
        )
        spike_ims, spike_axes = plot_spikes(
            {layer: spikes[layer].get("s").view(time, 1, -1) for layer in spikes},
            ims=spike_ims,
            axes=spike_axes,
        )
        weights_im = plot_weights(square_weights, im=weights_im)
        assigns_im = plot_assignments(square_assignments, im=assigns_im)
        perf_ax = plot_performance(accuracy, x_scale=update_interval, ax=perf_ax)
        voltage_ims, voltage_axes = plot_voltages(
            voltages, ims=voltage_ims, axes=voltage_axes
        )
        
        plt.pause(1e-8)
    
    # 重置网络状态变量
    network.reset_state_variables()
    pbar.set_description_str("Train progress: ")
    pbar.update()

print("Progress: %d / %d \n" % (n_train, n_train))
print("Training complete.\n")

print("Testing....\n")

# 测试准确率
accuracy = {"all": 0, "proportion": 0}

# 记录测试脉冲
spike_record = torch.zeros(1, time, n_neurons, device=device)

# 测试网络
print("\nBegin testing\n")
network.train(mode=False)

pbar = tqdm(total=n_test)
for i in range(n_test):
    # 获取当前样本
    x = X_test[i]
    label = y_test[i]
    
    # 将特征归一化到[0,1]范围，然后乘以强度
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8) * intensity
    
    # 编码为脉冲
    spike_train = encoder(torch.Tensor(x_norm))
    
    # 运行网络
    if gpu:
        inputs = {"X": spike_train.cuda().view(time, 1, 2, 2)}
    else:
        inputs = {"X": spike_train.view(time, 1, 2, 2)}
    
    network.run(inputs=inputs, time=time)
    
    # 记录脉冲
    spike_record[0] = spikes["Ae"].get("s").view(time, n_neurons)
    
    # 获取网络预测
    all_activity_pred = all_activity(
        spikes=spike_record, assignments=assignments, n_labels=n_classes
    )
    proportion_pred = proportion_weighting(
        spikes=spike_record,
        assignments=assignments,
        proportions=proportions,
        n_labels=n_classes,
    )
    
    # 计算准确率
    accuracy["all"] += float(torch.tensor(label).long() == all_activity_pred)
    accuracy["proportion"] += float(
        torch.tensor(label).long() == proportion_pred
    )
    
    # 重置网络状态变量
    network.reset_state_variables()
    
    pbar.set_description_str(
        f"Accuracy: {(max(accuracy['all'], accuracy['proportion']) / (i+1)):.3f}"
    )
    pbar.update()

print("\nAll activity accuracy: %.2f" % (100 * accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (100 * accuracy["proportion"] / n_test))

print("Testing complete.\n") 