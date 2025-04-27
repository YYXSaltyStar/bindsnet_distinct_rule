import bindsnet
import argparse
import os
from time import time as t

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
from bindsnet.encoding import PoissonEncoder, poisson
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import IncreasingInhibitionNetwork
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=5000)
parser.add_argument("--n_train", type=int, default=30000)
parser.add_argument("--n_workers", type=int, default=0)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=64)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--update_inhibation_weights", type=int, default=500)
parser.add_argument("--plot_interval", type=int, default=250)
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=True, gpu=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_epochs = args.n_epochs
n_test = args.n_test
n_train = args.n_train
n_workers = args.n_workers
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
plot_interval = args.plot_interval
update_interval = args.update_interval
plot = args.plot
gpu = args.gpu
#设定了以下一系列的参数命令行解释器，default代表了默认值。之后，我们可以在命令行中定义它们的值。
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_train", type=int, default=10000)
parser.add_argument("--n_test", type=int, default=1000)
parser.add_argument("--n_clamp", type=int, default=3)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=48)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--device_id", type=int, default=0)
parser.set_defaults(plot=True, gpu=True, train=True)#为三个刚才定义的自变量设置默认值。

args = parser.parse_args()#将之前设定的参数集合实例化

seed = args.seed#随机种子
n_neurons = args.n_neurons#神经元数量
n_train = args.n_train#训练集数量
n_test = args.n_test#测试集数量
n_clamp = args.n_clamp#每个样本强制激活的神经元的数量
exc = args.exc#兴奋性链接的初始强度
inh = args.inh#抑制性连接的初始强度
theta_plus = args.theta_plus#神经元发放脉冲后阈值的增量。
time = args.time#单次输入模拟的时间步长
dt = args.dt#单个时间步的持续时间（单位：毫秒）
intensity = args.intensity#调整泊松编码的脉冲频率，值越大，像素高亮度区域脉冲越密集。
progress_interval = args.progress_interval#进度条更新间隔
update_interval = args.update_interval
train = args.train#train和test共享一个布尔变量，train时为true，test时为false
plot = args.plot
gpu = args.gpu
device_id = args.device_id

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

from bindsnet.learning import LearningRule  # 引入基类

# 修改后的学习规则类
# 修改后的学习规则类
class SpatialLearningRule(LearningRule):
    def __init__(
        self,
        connection,  # 框架自动传入连接对象
        nu,          # 学习率
        input_shape=(28,28),
        threshold=0.2,
        **kwargs
    ):
        super().__init__(
            connection=connection,
            nu=nu,
            **kwargs
        )
        
        self.input_shape = input_shape
        self.threshold = threshold
        
        # 初始化卷积核（延迟设备设置）
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv.weight.data = torch.tensor(
            [[[[0.8,1.0,0.8],
               [1.0,0.0,1.0],
               [0.8,1.0,0.8]]]],
            dtype=torch.float32
        )
        self.conv.requires_grad_(False)
        self.device = None  # 延迟设备初始化

    def update(self, **kwargs):
        # 延迟设备配置
        if self.device is None:
            self.device = self.connection.w.device
            self.conv = self.conv.to(self.device)
        
        # 获取脉冲数据（添加维度检查）
        pre_spike = self.source.s.float()
        if pre_spike.dim() == 3:  # 处理时间维度
            pre_spike = pre_spike.mean(dim=(0,1))
        else:
            pre_spike = pre_spike.mean(dim=0)
            
        post_act = self.target.s.float().sum(dim=(0,1))  # (100,)
        
        # 空间共活性计算
        try:
            spike_2d = pre_spike.view(*self.input_shape).unsqueeze(0).unsqueeze(0)
            neighbor_act = self.conv(spike_2d).squeeze()
        except Exception as e:
            print(f"输入形状异常: {pre_spike.shape} -> 目标形状: {self.input_shape}")
            raise e
            
        spatial_boost = torch.sigmoid(neighbor_act * 3)
        
        # 突触后激活掩码
        post_mask = (post_act > self.threshold).float()
        
        # 计算权重增量（添加形状检查）
        delta_w = self.nu[0] * spatial_boost.flatten().unsqueeze(1) * post_mask.unsqueeze(0)
        if delta_w.shape != self.connection.w.shape:
            print(f"形状不匹配: delta_w {delta_w.shape} vs w {self.connection.w.shape}")
            delta_w = delta_w[:self.connection.w.size(0), :self.connection.w.size(1)]
        
        # 应用更新（添加设备同步）
        self.connection.w.data = (
            self.connection.w.data.to(self.device) + delta_w.to(self.device)
        ).clamp_(min=self.connection.wmin, max=self.connection.wmax)
        
input_shape = (28, 28)
learning_rate = 5e-4 

from bindsnet.network import Network
network = Network()

from bindsnet.network.nodes import Input, LIFNodes

input_layer = Input(
    n=784, 
    shape=(1,28,28),
    traces=True,  # 启用脉冲轨迹跟踪
    tc_trace=20.0  # 轨迹衰减时间常数[6](@ref)
)

ae_layer = LIFNodes(
    n=100,
    thresh=-52.0,       # 发放阈值（生物合理范围-55~-50mV）
    rest=-65.0,         # 静息电位
    tc_decay=100.0,     # 电压衰减时间常数（100ms）[7](@ref)
    refrac=5,           # 不应期时长（5ms）
    traces=True,        # 用于STDP学习
    theta_plus=0.05,    # 发放后阈值增量[7](@ref)
    noise_std=0.2       # 高斯噪声增强鲁棒性[3](@ref)
)

network.add_layer(input_layer, name="X")
network.add_layer(ae_layer, name="Ae")

from bindsnet.network.topology import Connection
from bindsnet.learning import PostPre

feedforward_conn = Connection(
    source=input_layer,
    target=ae_layer,
    w=0.05 + 0.1 * torch.randn(784, 100),
    wmin=0.0,
    wmax=100.0,
    update_rule=SpatialLearningRule,
    nu=(5e-4, 0),      # 仅使用前项学习率
    input_shape=(28,28),
    threshold=0.2,
    reduction=torch.sum,
    weight_decay=0.0   # 明确设置权重衰减
)


print("学习规则绑定的连接:", feedforward_conn.update_rule.connection is feedforward_conn)
print("权重矩阵形状:", feedforward_conn.w.shape)
print("权重设备:", feedforward_conn.w.device)
print("卷积核设备:", feedforward_conn.update_rule.conv.weight.device)

recurrent_conn = Connection(
    source=ae_layer,
    target=ae_layer,
    update_rule=PostPre,  # 启用STDP
    nu=(1e-5, 1e-4),      # 前/后事件学习率（需低于前馈连接）
    w=0.025*(torch.eye(100)-1),
    wmin=-120.0,
    wmax=0.0,             # 维持抑制性连接特性
    tc_trace=20.0         # 脉冲轨迹衰减时间常数[6](@ref)
)

network.add_connection(feedforward_conn, "X", "Ae")
network.add_connection(recurrent_conn, "Ae", "Ae")

from bindsnet.network.monitors import Monitor

# 电压监控（Ae层）
voltage_monitor = Monitor(
    obj=ae_layer,
    state_vars=["v"],  # 监控膜电位
    time=250,          # 记录250个时间步
    device=device      # GPU/CPU设备
)


# GPU加速配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    input_layer.to(device)
    ae_layer.to(device)


# 动态抑制强度调整（训练过程中增强）
def update_inhibition(step, max_steps=10000):
    inhib = -40 * (step / max_steps)  # 从0线性增强到-40
    recurrent_conn.w.data = torch.clamp(
        recurrent_conn.w.data * 0.9 + inhib * 0.1,
        min=-120.0, max=0.0
    )

dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)

n_classes = 10
assignments = -torch.ones(n_neurons, device=device)
proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((n_neurons, n_classes), device=device)

accuracy = {"all": [], "proportion": []}

som_voltage_monitor = Monitor(
    network.layers["Ae"], ["v"], time=int(time / dt), device=device
)
network.add_monitor(som_voltage_monitor, name="som_voltage")

spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(
        network.layers[layer], state_vars=["v"], time=int(time / dt), device=device
    )
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None
save_weights_fn = "plots/weights/weights.png"
save_performance_fn = "plots/performance/performance.png"
save_assaiments_fn = "plots/assaiments/assaiments.png"

directorys = ["plots", "plots/weights", "plots/performance", "plots/assaiments"]
for directory in directorys:
    if not os.path.exists(directory):
        os.makedirs(directory)

print("\nBegin training.\n")
start = t()

for epoch in range(n_epochs):
    labels = []

    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    # Create a dataloader to iterate and batch data
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=n_workers, pin_memory=gpu
    )

    pbar = tqdm(total=n_train)
    for step, batch in enumerate(dataloader):
        if step == n_train:
            break

        # Get next input sample.
        inputs = {
            "X": batch["encoded_image"].view(int(time / dt), 1, 1, 28, 28).to(device)
        }

        if step > 0 and step % update_interval == 0 :
            label_tensor = torch.tensor(labels, device=device)
            all_activity_pred = all_activity(
                spikes=spike_record, assignments=assignments, n_labels=n_classes
            )
            proportion_pred = proportion_weighting(
                spikes=spike_record,
                assignments=assignments,
                proportions=proportions,
                n_labels=n_classes,
            )
            accuracy["all"].append(
                100
                * torch.sum(label_tensor.long() == all_activity_pred).item()
                / len(label_tensor)
            )
            accuracy["proportion"].append(
                100
                * torch.sum(label_tensor.long() == proportion_pred).item()
                / len(label_tensor)
            )

            tqdm.write(
                "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                % (
                    accuracy["all"][-1],
                    np.mean(accuracy["all"]),
                    np.max(accuracy["all"]),
                )
            )
            tqdm.write(
                "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f"
                " (best)\n"
                % (
                    accuracy["proportion"][-1],
                    np.mean(accuracy["proportion"]),
                    np.max(accuracy["proportion"]),
                )
            )

            # Assign labels to excitatory layer neurons.
            assignments, proportions, rates = assign_labels(
                spikes=spike_record,
                labels=label_tensor,
                n_labels=n_classes,
                rates=rates,
            )

            labels = []

        labels.append(batch["label"])

        temp_spikes = 0
        factor = 1.2
        for retry in range(5):
            # Run the network on the input.
            network.run(inputs=inputs, time=time)

            # Get spikes from the network
            temp_spikes = spikes["Y"].get("s").squeeze()

            if temp_spikes.sum().sum() < 2:
                inputs["X"] *= (
                    poisson(
                        datum=factor * batch["image"].clamp(min=0),
                        dt=dt,
                        time=int(time / dt),
                    )
                    .to(device)
                    .view(int(time / dt), 1, 1, 28, 28)
                )
                factor *= factor
            else:
                break

        # Get voltage recording.
        exc_voltages = som_voltage_monitor.get("v")

        # Add to spikes recording.
        # spike_record[step % update_interval] = temp_spikes.detach().clone().cpu()
        spike_record[step % update_interval].copy_(temp_spikes, non_blocking=True)

        # Optionally plot various simulation information.
        if plot and step % plot_interval == 0:
            image = batch["image"].view(28, 28)
            inpt = inputs["X"].view(time, 784).sum(0).view(28, 28)
            input_exc_weights = network.connections[("X", "Y")].w
            square_weights = get_square_weights(
                input_exc_weights.view(784, n_neurons), n_sqrt, 28
            )
            square_assignments = get_square_assignments(assignments, n_sqrt)
            spikes_ = {layer: spikes[layer].get("s") for layer in spikes}
            voltages = {"Y": exc_voltages}
            inpt_axes, inpt_ims = plot_input(
                image, inpt, label=batch["label"], axes=inpt_axes, ims=inpt_ims
            )
            spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
            [weights_im, save_weights_fn] = plot_weights(
                square_weights, im=weights_im, save=save_weights_fn
            )
            assigns_im = plot_assignments(
                square_assignments, im=assigns_im, save=save_assaiments_fn
            )
            perf_ax = plot_performance(accuracy, ax=perf_ax, save=save_performance_fn)
            voltage_ims, voltage_axes = plot_voltages(
                voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
            )
            #
            plt.pause(1e-8)

        network.reset_state_variables()  # Reset state variables.
        pbar.set_description_str("Train progress: ")
        pbar.update()

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Training complete.\n")

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

# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Record spikes during the simulation.
spike_record = torch.zeros(1, int(time / dt), n_neurons, device=device)

# Train the network.
print("\nBegin testing\n")
network.train(mode=False)
start = t()

pbar = tqdm(total=n_test)
for step, batch in enumerate(test_dataset):
    if step >= n_test:
        break
    # Get next input sample.
    inputs = {"X": batch["encoded_image"].view(int(time / dt), 1, 1, 28, 28)}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    network.run(inputs=inputs, time=time)

    # Add to spikes recording.
    spike_record[0] = spikes["Y"].get("s").squeeze()

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(batch["label"], device=device)

    # Get network predictions.
    all_activity_pred = all_activity(
        spikes=spike_record, assignments=assignments, n_labels=n_classes
    )
    proportion_pred = proportion_weighting(
        spikes=spike_record,
        assignments=assignments,
        proportions=proportions,
        n_labels=n_classes,
    )

    # Compute network accuracy according to available classification strategies.
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
    accuracy["proportion"] += float(
        torch.sum(label_tensor.long() == proportion_pred).item()
    )

    network.reset_state_variables()  # Reset state variables.
    pbar.set_description_str("Test progress: ")
    pbar.update()

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Testing complete.\n")