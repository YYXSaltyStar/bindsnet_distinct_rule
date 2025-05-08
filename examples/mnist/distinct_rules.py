import argparse
import os
from time import time as t
import matplotlib.pyplot as plt
from bindsnet.learning import LearningRule
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.learning import PostPre
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
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights

# 参数配置
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=1000)
parser.add_argument("--n_train", type=int, default=10000)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=48)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()

# 设置随机种子
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.set_num_threads(os.cpu_count() - 1)

# 网络组件
class SpatialLearningRule:
    def __init__(self, connection, nu, input_shape=(28,28), threshold=0.2):
        self.connection = connection
        self.nu = nu
        self.input_shape = input_shape
        self.threshold = threshold
        
        # 固定卷积核
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv.weight.data = torch.tensor(
            [[[[0.8,1.0,0.8],
               [1.0,0.0,1.0],
               [0.8,1.0,0.8]]]],
            dtype=torch.float32
        )
        self.conv.requires_grad_(False)

    def update(self):
        pre_spike = self.connection.source.s.float().mean(dim=(0,1))
        post_act = self.connection.target.s.float().sum(dim=(0,1))
        
        try:
            spike_2d = pre_spike.view(*self.input_shape).unsqueeze(0).unsqueeze(0)
            neighbor_act = self.conv(spike_2d).squeeze()
        except Exception as e:
            print(f"Shape error: {pre_spike.shape} -> {self.input_shape}")
            raise e
            
        spatial_boost = torch.sigmoid(neighbor_act * 3)
        post_mask = (post_act > self.threshold).float()
        
        delta_w = self.nu[0] * spatial_boost.flatten().unsqueeze(1) * post_mask.unsqueeze(0)
        self.connection.w += delta_w
        self.connection.w.clamp_(min=0.0, max=100.0)

# 初始化网络
network = Network()

# 输入层
input_layer = Input(n=784, shape=(1,28,28), traces=True, tc_trace=20.0)
network.add_layer(input_layer, name="X")

# 神经元层
ae_layer = LIFNodes(
    n=args.n_neurons,
    thresh=-52.0,
    rest=-65.0,
    tc_decay=100.0,
    refrac=5,
    theta_plus=args.theta_plus
)
network.add_layer(ae_layer, name="Ae")

# 前馈连接
feedforward_conn = Connection(
    source=input_layer,
    target=ae_layer,
    w=0.05 + 0.1 * torch.randn(784, args.n_neurons),
    wmin=0.0,
    wmax=100.0,
    update_rule=SpatialLearningRule,
    nu=(5e-4, 0)
)
network.add_connection(feedforward_conn, "X", "Ae")

# 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * args.intensity)
])

dataset = MNIST(
    PoissonEncoder(time=args.time, dt=args.dt),
    None,
    root=os.path.join("data", "MNIST"),
    download=True,
    transform=transform
)

# 训练准备
spike_record = torch.zeros((args.update_interval, int(args.time/args.dt), args.n_neurons))
assignments = -torch.ones(args.n_neurons)
proportions = torch.zeros((args.n_neurons, 10))
accuracy = {"all": [], "proportion": []}

# 训练循环
print("\nBegin training")
for epoch in range(args.n_epochs):
    labels = []
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    pbar = tqdm(total=args.n_train, desc=f"Epoch {epoch+1}")
    for step, batch in enumerate(dataloader):
        if step >= args.n_train:
            break
            
        # 输入处理
        inputs = {
            "X": batch["encoded_image"].view(int(args.time/args.dt), 1, 1, 28, 28)
        }
        
        # 网络运行
        network.run(inputs=inputs, time=args.time)
        
        # 记录数据
        spike_record[step % args.update_interval] = network.monitors["Ae_spikes"].get("s").squeeze()
        labels.append(batch["label"])
        
        # 定期更新
        if step > 0 and step % args.update_interval == 0:
            label_tensor = torch.tensor(labels)
            
            # 计算准确率
            all_pred = all_activity(spike_record, assignments, 10)
            prop_pred = proportion_weighting(spike_record, assignments, proportions, 10)
            
            accuracy["all"].append(100 * (label_tensor == all_pred).float().mean().item())
            accuracy["proportion"].append(100 * (label_tensor == prop_pred).float().mean().item())
            
            # 更新标签分配
            assignments, proportions, _ = assign_labels(spike_record, label_tensor, 10)
            labels = []
        
        network.reset_state_variables()
        pbar.update(1)
    pbar.close()

print("Training complete\n")

# 测试流程
test_dataset = MNIST(
    PoissonEncoder(time=args.time, dt=args.dt),
    None,
    train=False,
    transform=transform
)

correct_all = 0
correct_prop = 0
print("\nBegin testing")
with torch.no_grad():
    pbar = tqdm(total=args.n_test)
    for i, batch in enumerate(test_dataset):
        if i >= args.n_test:
            break
            
        inputs = {"X": batch["encoded_image"].view(int(args.time/args.dt), 1, 1, 28, 28)}
        network.run(inputs=inputs, time=args.time)
        
        # 获取预测结果
        spikes = network.monitors["Ae_spikes"].get("s").squeeze().unsqueeze(0)
        label = batch["label"]
        
        all_pred = all_activity(spikes, assignments, 10)
        prop_pred = proportion_weighting(spikes, assignments, proportions, 10)
        
        correct_all += int(all_pred.item() == label)
        correct_prop += int(prop_pred.item() == label)
        
        network.reset_state_variables()
        pbar.update(1)
    pbar.close()

print(f"\nFinal Accuracy:")
print(f"All activity: {100 * correct_all / args.n_test:.2f}%")
print(f"Proportion weighting: {100 * correct_prop / args.n_test:.2f}%")
print("Testing complete")