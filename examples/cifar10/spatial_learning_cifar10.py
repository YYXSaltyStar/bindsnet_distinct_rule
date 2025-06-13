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
from bindsnet.learning import LearningRule

class SpatialLearningRule(LearningRule):
    def __init__(self, connection, nu, **kwargs):
        super().__init__(connection=connection, nu=nu)
        self.device = connection.source.s.device
        self.input_shape = kwargs.get('input_shape', (28, 28))
        self.output_shape = kwargs.get('output_shape', (10, 10))  # 新增输出层形状
        self.threshold = kwargs.get('threshold', 0.15)
        self.window = kwargs.get('window', 10)
        self.neighbor_radius = kwargs.get('neighbor_radius', 1)
        self.tau = kwargs.get('tau', 1.0)

    def map_coordinates(self, pre_y, pre_x):
        """将输入层坐标映射到输出层"""
        scale_y = self.output_shape[0] / self.input_shape[0]
        scale_x = self.output_shape[1] / self.input_shape[1]
        post_y = int(pre_y * scale_y)
        post_x = int(pre_x * scale_x)
        return post_y, post_x

    def get_region_coordinates(self, pre_y, pre_x):
        """获取输入层中单个神经元，对应输出层中多个神经元的区域坐标"""
        scale_y = self.input_shape[0] / self.output_shape[0]
        scale_x = self.input_shape[1] / self.output_shape[1]
        
        # 计算区域范围（对于输出层的一个单点）
        start_y = int(pre_y * scale_y)
        start_x = int(pre_x * scale_x)
        end_y = int((pre_y + 1) * scale_y)
        end_x = int((pre_x + 1) * scale_x)
        
        # 生成区域内的所有坐标
        coords = []
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                if 0 <= y < self.input_shape[0] and 0 <= x < self.input_shape[1]:
                    coords.append((y, x))
        return coords

    def get_region_spikes(self, pre_spikes, region_coords, t, window):
        """获取输入层中区域内的脉冲活动"""
        region_spikes = []
        for y, x in region_coords:
            idx = y * self.input_shape[1] + x
            spikes = pre_spikes[t:t+window, idx]
            region_spikes.append(spikes)
        return torch.stack(region_spikes).max(dim=0)[0]  # 取最大值

    def update(self, **kwargs):
        pre_spikes = self.connection.source.s.float().to(self.device)
        post_spikes = self.connection.target.s.float().to(self.device)
        
        # 重塑张量维度
        if len(pre_spikes.shape) == 4:  # 例如, 输入层脉冲可能是 [batch_size, channels, height, width] -> [1, 1, 32, 32]
            pre_spikes = pre_spikes.view(pre_spikes.shape[0], pre_spikes.shape[1], -1)  # 重塑为 [1, 1, 32*32]
        if len(post_spikes.shape) == 4:
            post_spikes = post_spikes.view(1, 1, -1)
        
        # 检查并调整维度
        if len(pre_spikes.shape) == 2:  # 如果只有两个维度 (T, N)
            pre_spikes = pre_spikes.unsqueeze(1)  # 添加批次维度 (T, 1, N)
        if len(post_spikes.shape) == 2:
            post_spikes = post_spikes.unsqueeze(1)
        
        T, B, N_pre = pre_spikes.shape
        _, _, N_post = post_spikes.shape
        pre_spikes = pre_spikes[:, 0, :]
        post_spikes = post_spikes[:, 0, :]

        # 确保所有权重相关张量在正确的设备上
        w = self.connection.w.data.to(self.device)
        wmin = torch.tensor(self.connection.wmin, device=self.device)
        wmax = torch.tensor(self.connection.wmax, device=self.device)
        delta_w = torch.zeros_like(w, device=self.device)
        
        # 遍历所有时间步
        for t in range(T - self.window):  # 确保有足够的窗口大小
            # 遍历所有连接
            for i in range(N_pre):
                for j in range(N_post):
                    # 获取时间窗口内的脉冲
                    pre_window = pre_spikes[t:t+self.window, i]
                    post_window = post_spikes[t:t+self.window, j]
                    
                    # STDP学习 - 使用原生方式
                    pre_times = (pre_window > 0).nonzero(as_tuple=True)[0]
                    post_times = (post_window > 0).nonzero(as_tuple=True)[0]
                    
                    if len(pre_times) > 0 and len(post_times) > 0:
                        # 计算所有可能的时间差
                        time_diffs = post_times.unsqueeze(0) - pre_times.unsqueeze(1)
                        # 只保留前神经元先发放的情况
                        valid_diffs = time_diffs[time_diffs > 0]
                        
                        if len(valid_diffs) > 0:
                            # 计算权重更新量
                            updates = self.nu[0] * torch.exp(-valid_diffs / self.tau)
                            # 累加所有有效的更新量
                            delta_w[i, j] += updates.sum()
                    
                    # 空间邻域增强 - 保留空间映射功能
                    pre_y, pre_x = divmod(i, self.input_shape[1])
                    for dy in range(-self.neighbor_radius, self.neighbor_radius+1):
                        for dx in range(-self.neighbor_radius, self.neighbor_radius+1):
                            ni, nj = pre_y+dy, pre_x+dx
                            if 0 <= ni < self.input_shape[0] and 0 <= nj < self.input_shape[1]:
                                n_idx = ni * self.input_shape[1] + nj
                                if n_idx < N_pre:
                                    # 获取邻域区域的脉冲
                                    neighbor_region_coords = self.get_region_coordinates(
                                        self.map_coordinates(ni, nj)[0],
                                        self.map_coordinates(ni, nj)[1]
                                    )
                                    neighbor_pre_spikes = self.get_region_spikes(
                                        pre_spikes, neighbor_region_coords, t, self.window
                                    )
                                    
                                    if len((neighbor_pre_spikes > 0).nonzero(as_tuple=True)[0]) > 0 and \
                                       len(post_times) > 0:
                                        # 邻域增强
                                        for y, x in neighbor_region_coords:
                                            idx = y * self.input_shape[1] + x
                                            delta_w[idx, j] += self.nu[0] * 0.5 / len(neighbor_region_coords)

        # 权重更新
        w += delta_w
        w.clamp_(wmin, wmax)
        self.connection.w.data.copy_(w)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_train", type=int, default=50000)  # CIFAR10 has 50k training images
parser.add_argument("--n_test", type=int, default=10000)
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
parser.set_defaults(plot=False, gpu=True, train=True)

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
    update_rule=SpatialLearningRule,
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
    root=os.path.join("..", "..", "data", "CIFAR10"), # Changed path
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
    root=os.path.join("..", "..", "data", "CIFAR10"), # Changed path
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

        network.reset_state_variables()
        
        # 更新进度
        current_samples += 1
        pbar.update(1)

    pbar.set_description_str(
        f"准确率: {(max(accuracy['all'] ,accuracy['proportion'] ) / current_samples):.3}"
    )

print("\nAll activity准确率: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting准确率: %.2f \n" % (accuracy["proportion"] / n_test))

print("测试完成.\n")