import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from bindsnet.network.topology import Connection
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
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights
from bindsnet.learning import PostPre


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_train", type=int, default=1000)  # CIFAR10 has 50k training images
parser.add_argument("--n_test", type=int, default=1000)
parser.add_argument("--n_clamp", type=int, default=1)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=32) # May need adjustment for CIFAR10
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

device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
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
n_sqrt = int(np.ceil(np.sqrt(n_neurons))) # For output layer visualization
start_intensity = intensity
per_class = int(n_neurons / n_classes)

# 创建网络对象
torch.manual_seed(seed)
network = Network(dt=dt)

# Input layer for CIFAR10 (grayscale 32x32)
input_layer = Input(n=32*32, shape=(1, 32, 32), traces=True)
# 兴奋层
exc_layer = LIFNodes(n=n_neurons, traces=True, theta_plus=theta_plus)
# 抑制层
inh_layer = LIFNodes(n=n_neurons, traces=True)

# 添加层到网络
network.add_layer(input_layer, name="X")
network.add_layer(exc_layer, name="E")
network.add_layer(inh_layer, name="I")

# 输入层到兴奋层的连接（可学习）
input_exc_conn = Connection(
    source=input_layer,
    target=exc_layer,
    w=0.3 * torch.rand(32*32, n_neurons), # Adjusted for 32x32 input
    update_rule=PostPre,
    nu=1e-2,
    wmin=0.0,
    wmax=1.0,
    input_shape=(32, 32), # Adjusted for 32x32 input
    output_shape=(int(np.sqrt(n_neurons)), int(np.sqrt(n_neurons))), # e.g. (10,10) if n_neurons=100
    window=10,
    neighbor_radius=1,
    tau=1.0
)
network.add_connection(input_exc_conn, source="X", target="E")

# 兴奋层到抑制层的连接（固定强抑制）
exc_inh_conn = Connection(
    source=exc_layer,
    target=inh_layer,
    w=exc * torch.eye(n_neurons),
    update_rule=None,
)
network.add_connection(exc_inh_conn, source="E", target="I")

# 抑制层到兴奋层的连接（全连接，固定强抑制）
inh_exc_conn = Connection(
    source=inh_layer,
    target=exc_layer,
    w=-inh * (torch.ones(n_neurons, n_neurons) - torch.eye(n_neurons)),
    update_rule=None,
)
network.add_connection(inh_exc_conn, source="I", target="E")

if gpu:
    network.to("cuda")

# 创建数据加载器
from bindsnet.datasets import CIFAR10 # Changed from MNIST
dataset = CIFAR10(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("data", "CIFAR10"), # Changed path
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(), # Convert to grayscale for SpatialLearningRule
            transforms.Lambda(lambda x: x * intensity)
        ]
    ),
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

# 添加监视器
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)
    
# 初始化权重绘图相关变量 
if plot:
    # 确保保存权重的目录存在
    if not os.path.exists('./weight_result'):
        os.makedirs('./weight_result')
    if not os.path.exists('./weight_result/STDP_cifar10'):
        os.makedirs('./weight_result/STDP_cifar10')

# 训练网络
print("开始训练 (CIFAR10 Grayscale)...")


# 计算实际的训练步数
n_steps = (n_train + batch_size - 1) // batch_size
pbar = tqdm(total=n_train)
current_samples = 0

for i, datum in enumerate(dataloader):
    if current_samples >= n_train:
        break

    images = datum["encoded_image"]  # [batch_size, time, 1, 32, 32] for grayscale
    labels_batch = datum["label"]    # [batch_size]

    # 处理每个批次中的样本
    for b in range(batch_size):
        if current_samples >= n_train:
            break
            
        # 添加当前标签
        labels[current_samples % update_interval] = labels_batch[b]

        # 运行网络
        choice = np.random.choice(int(n_neurons / n_classes), size=n_clamp, replace=False)
        clamp = {"E": per_class * labels_batch[b].long() + torch.Tensor(choice).long()}
        
        if gpu:
            inputs = {"X": images[b].cuda().view(time, 1, 1, 32, 32)} # Adjusted for 32x32
        else:
            inputs = {"X": images[b].view(time, 1, 1, 32, 32)} # Adjusted for 32x32
            
        network.run(inputs=inputs, time=time, clamp=clamp)

        # 记录脉冲
        spike_record[current_samples % update_interval] = spikes["E"].get("s").view(time, n_neurons)

        # 只绘制权重 - 每隔250个样本保存一次
        if plot and current_samples % 250 == 0:  # 每250个样本绘制和保存权重
            # 获取权重并生成正方形权重图
            input_exc_weights = network.connections[("X", "E")].w
            square_weights = get_square_weights(input_exc_weights.view(32*32, n_neurons), n_sqrt, 32)
            
            # 绘制权重
            plt.figure(figsize=(10, 10))
            weights_im = plot_weights(square_weights, im=None)
            plt.title(f"训练权重 - 样本 {current_samples}")
            
            # 确保目录存在
            if not os.path.exists('./weight_result/STDP_cifar10'):
                os.makedirs('./weight_result/STDP_cifar10')
                
            # 保存图像
            plt.savefig(f'./weight_result/STDP_cifar10/weights_sample_{current_samples}.png')
            plt.close()

        # 重置网络状态
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

print("开始测试 (CIFAR10 Grayscale")

# 加载测试数据
test_dataset = CIFAR10( # Changed from MNIST
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("data", "CIFAR10"), # Changed path
    download=True,
    train=False,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(), # Convert to grayscale
            transforms.Lambda(lambda x: x * intensity)
        ]
    ),
)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 测试准确率
accuracy = {"all": 0, "proportion": 0}

# 记录测试脉冲
spike_record = torch.zeros(batch_size, int(time / dt), n_neurons, device=device)

# 测试网络
network.train(mode=False)

pbar = tqdm(total=n_test)
current_samples = 0

for step, batch in enumerate(test_dataloader):
    if current_samples >= n_test:
        break
    
    images = batch["encoded_image"]  # [batch_size, time, 1, 32, 32] for grayscale
    labels_batch = batch["label"]    # [batch_size]
    
    # 处理每个批次中的样本
    for b in range(batch_size):
        if current_samples >= n_test:
            break
            
        # 获取输入样本
        if gpu:
            inputs = {"X": images[b].cuda().view(int(time / dt), 1, 1, 32, 32)} # Adjusted for 32x32
        else:
            inputs = {"X": images[b].view(int(time / dt), 1, 1, 32, 32)} # Adjusted for 32x32

        # 运行网络
        network.run(inputs=inputs, time=time) # No clamp during testing

        # 记录脉冲
        spike_record[b] = spikes["E"].get("s").squeeze()

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

        # 在测试阶段不进行绘图
            
        network.reset_state_variables()
        
        # 更新进度
        current_samples += 1
        pbar.update(1)

    pbar.set_description_str(
        f"准确率: {(max(accuracy['all'] ,accuracy['proportion'] ) / current_samples):.3}"
    )

print("\nAll activity准确率: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting准确率: %.2f \n" % (accuracy["proportion"] / n_test))


# 可视化最终结果
if plot:
    # 绘制最终权重和神经元分配
    input_exc_weights = network.connections[("X", "Ae")].w
    square_weights = get_square_weights(input_exc_weights.view(3*32*32, n_neurons), n_sqrt, 32)
    square_assignments = get_square_assignments(assignments, n_sqrt)
    
    plt.figure(figsize=(12, 10))
    
    # 绘制最终权重
    plt.subplot(2, 2, 1)
    weights_im = plot_weights(square_weights, im=None)
    plt.title("最终学习权重")
    
    # 绘制神经元分配
    plt.subplot(2, 2, 2)
    assigns_im = plot_assignments(square_assignments, im=None)
    plt.title("神经元类别分配")
    
    # 绘制测试准确率
    plt.subplot(2, 2, 3)
    plt.bar(['All Activity', 'Proportion'], [accuracy["all"] / n_test, accuracy["proportion"] / n_test])
    plt.ylim([0, 100])
    plt.title("测试准确率")
    
    # 保存最终结果
    plt.tight_layout()
    if not os.path.exists('./weight_result/supervised_cifar10'):
        os.makedirs('./weight_result/supervised_cifar10')
    plt.savefig('./weight_result/supervised_cifar10/final_weights.png')
    plt.show()

print("Testing complete.\n") 