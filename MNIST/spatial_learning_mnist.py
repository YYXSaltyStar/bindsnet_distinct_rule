import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from my_SPL_tool import DiehlAndCook2015_with_SPL
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_train", type=int, default=1000)
parser.add_argument("--n_test", type=int, default=1000)
parser.add_argument("--n_clamp", type=int, default=1)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=32)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=32)
parser.set_defaults(plot=True, gpu=True, train=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_train = args.n_train
n_test = args.n_test
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
batch_size = args.batch_size

# Sets up Gpu use
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

if not train:
    update_interval = n_test

n_classes = 10
n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity
per_class = int(n_neurons / n_classes)

# Build Diehl & Cook 2015 network.
network = DiehlAndCook2015_with_SPL(
    n_inpt=784,
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    dt=dt,
    nu=[1e-10, 1e-3],  # 0.711
    norm=78.4,
    theta_plus=theta_plus,
    inpt_shape=(1, 28, 28),
)

# Directs network to GPU
if gpu:
    network.to("cuda")

# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(network.layers["Ae"], ["v"], time=time, device=device)
inh_voltage_monitor = Monitor(network.layers["Ai"], ["v"], time=time, device=device)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")

# Load MNIST data.
dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Create a dataloader to iterate and batch data
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Record spikes during the simulation.
spike_record = torch.zeros(update_interval, time, n_neurons, device=device)

# Neuron assignments and spike proportions.
assignments = -torch.ones_like(torch.Tensor(n_neurons), device=device)
proportions = torch.zeros_like(torch.Tensor(n_neurons, n_classes), device=device)
rates = torch.zeros_like(torch.Tensor(n_neurons, n_classes), device=device)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

# Labels to determine neuron assignments and spike proportions and estimate accuracy
labels = torch.empty(update_interval, device=device)

spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

# Train the network.
print("Begin training.\n")

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
current_samples = 0

for i, datum in enumerate(dataloader):
    if current_samples >= n_train:
        break

    images = datum["encoded_image"]  # [batch_size, time, 1, 28, 28]
    labels_batch = datum["label"]    # [batch_size]

    # 处理每个批次中的样本
    for b in range(batch_size):
        if current_samples >= n_train:
            break

        # 添加当前标签
        labels[current_samples % update_interval] = labels_batch[b]

        # 运行网络
        choice = np.random.choice(int(n_neurons / n_classes), size=n_clamp, replace=False)
        clamp = {"Ae": per_class * labels_batch[b].long() + torch.Tensor(choice).long()}
        
        if gpu:
            inputs = {"X": images[b].cuda().view(time, 1, 1, 28, 28)}
        else:
            inputs = {"X": images[b].view(time, 1, 1, 28, 28)}
            
        network.run(inputs=inputs, time=time, clamp=clamp)

        # 获取电压记录
        exc_voltages = exc_voltage_monitor.get("v")
        inh_voltages = inh_voltage_monitor.get("v")

        # 记录脉冲
        spike_record[current_samples % update_interval] = spikes["Ae"].get("s").view(time, n_neurons)

        # 可选地绘制各种模拟信息
        if plot:
            input_exc_weights = network.connections[("X", "Ae")].w
            square_weights = get_square_weights(
                input_exc_weights.view(784, n_neurons), n_sqrt, 28
            )         
            weights_im = plot_weights(square_weights, im=weights_im)#图3 
            if plot and current_samples % 250 == 0:
                plt.savefig(f'./weight_result/spatial_learning_mnist/weights_sample_{current_samples}.png')   
            plt.pause(1e-8)

        # 重置网络状态变量
        network.reset_state_variables()
        
        # 更新进度
        current_samples += 1
        pbar.update(1)

        # 在每个update_interval后计算准确率
        if current_samples % update_interval == 0 and current_samples > 0:
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
                "\nAll activity准确率: %.2f (最新), %.2f (平均), %.2f (最佳)"
                % (accuracy["all"][-1], np.mean(accuracy["all"]), np.max(accuracy["all"]))
            )
            print(
                "Proportion weighting准确率: %.2f (最新), %.2f (平均), %.2f (最佳)\n"
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

    pbar.set_description_str(f"训练进度: {current_samples}/{n_train}")

print(f"\n进度: {current_samples} / {n_train} \n")
print("训练完成.\n")

print("开始测试....\n")

# Load MNIST data.
test_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# 测试数据加载器
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 准确率记录
accuracy = {"all": 0, "proportion": 0}

# 记录测试脉冲
spike_record = torch.zeros(batch_size, int(time / dt), n_neurons, device=device)

# 测试网络
print("\n开始测试\n")
network.train(mode=False)

pbar = tqdm(total=n_test)
current_samples = 0

for step, batch in enumerate(test_dataloader):
    if current_samples >= n_test:
        break
    
    images = batch["encoded_image"]  # [batch_size, time, 1, 28, 28]
    labels_batch = batch["label"]    # [batch_size]
    
    # 处理每个批次中的样本
    for b in range(batch_size):
        if current_samples >= n_test:
            break
            
        # 获取输入样本
        if gpu:
            inputs = {"X": images[b].cuda().view(int(time / dt), 1, 1, 28, 28)}
        else:
            inputs = {"X": images[b].view(int(time / dt), 1, 1, 28, 28)}

        # 运行网络
        network.run(inputs=inputs, time=time)

        # 记录脉冲
        spike_record[b] = spikes["Ae"].get("s").squeeze()

        # 获取网络预测
        all_activity_pred = all_activity(
            spikes=spike_record[b:b+1], assignments=assignments, n_labels=n_classes
        )
        proportion_pred = proportion_weighting(
            spikes=spike_record[b:b+1],
            assignments=assignments,
            proportions=proportions,
            n_labels=n_classes,
        )

        # 计算准确率
        accuracy["all"] += float(torch.sum(labels_batch[b].long() == all_activity_pred).item())
        accuracy["proportion"] += float(
            torch.sum(labels_batch[b].long() == proportion_pred).item()
        )

        network.reset_state_variables()  # 重置状态变量
        
        # 更新进度
        current_samples += 1
        pbar.update(1)

    pbar.set_description_str(
        f"准确率: {(max(accuracy['all'] ,accuracy['proportion'] ) / current_samples):.3}"
    )

print("\nAll activity准确率: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting准确率: %.2f \n" % (accuracy["proportion"] / n_test))

# 保存最终权重
if plot:
    # 获取最终权重
    input_exc_weights = network.connections[("X", "Ae")].w
    square_weights = get_square_weights(
        input_exc_weights.view(784, n_neurons), n_sqrt, 28
    )
    
    # 绘制最终权重
    plt.figure(figsize=(10, 10))
    weights_im = plot_weights(square_weights, im=None)
    plt.title("最终学习权重")
    
    # 确保目录存在
    if not os.path.exists('./weight_result/spatial_learning_mnist'):
        os.makedirs('./weight_result/spatial_learning_mnist')
        
    # 保存最终权重图像
    plt.savefig('./weight_result/spatial_learning_mnist/final_weights.png')
    plt.close()

print("Testing complete.\n")