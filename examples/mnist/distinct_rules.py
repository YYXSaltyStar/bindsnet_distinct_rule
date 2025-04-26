import argparse
import os
from bindsnet.network.nodes import Input, LIFNodes  # 导入神经元层类型
from bindsnet.network.topology import Connection    # 导入连接类型（用于类型校验）
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
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
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights
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
parser.add_argument("--apical_radius", type=int, default=20, help="顶突触空间邻域半径（像素）")
parser.add_argument("--early_phase", type=int, default=5000, help="早期阶段样本数")
parser.add_argument("--apical_lr", type=float, nargs=2, default=[0.01, 0.001], 
                   help="顶突触两阶段学习率")
args, _ = parser.parse_known_args()  # 正确获取已定义参数
parser.set_defaults(plot=True, gpu=True, train=True)
args = parser.parse_args()  # 最终解析

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

def get_layer_name(network, layer_obj):
    """通过层对象获取其在网络中的注册名称"""
    for name, layer in network.layers.items():
        if layer == layer_obj:  # 对象实例匹配
            return name
    return None

def get_neighborhood_mask(shape, radius=5):
    """生成空间邻近掩膜矩阵（示例：输入层28x28）"""
    h, w = shape
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
    dist = torch.sqrt((x[:,:,None,None] - x[None,None,:,:])**2 + 
                     (y[:,:,None,None] - y[None,None,:,:])**2)
    return (dist <= radius).float()

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
# 在定义n_sqrt后添加位置坐标生成
n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
positions = np.array([[i // n_sqrt, i % n_sqrt] for i in range(n_neurons)], dtype=np.float32)

# 如果需要显示输入层的位置（针对X_to_Ae连接）
input_positions = np.array([[i//28, i%28] for i in range(784)], dtype=np.float32)
start_intensity = intensity
per_class = int(n_neurons / n_classes)

# Build Diehl & Cook 2015 network.
network = DiehlAndCook2015(
    n_inpt=784,
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    dt=dt,
    nu=[1e-10, 1e-3],
    norm=78.4,
    theta_plus=theta_plus,
    inpt_shape=(1, 28, 28)  # 保留原始参数
)

print("网络层信息:")
for layer_name in network.layers:
    layer = network.layers[layer_name]
    print(f"{layer_name}: {type(layer)}")

n_sqrt = int(np.ceil(np.sqrt(n_neurons))) 

assert "Ae" in network.layers, "Ae层未正确定义"
assert "X" in network.layers, "输入层未正确定义"

def is_apical_connection(conn):
    """判断连接是否为顶突触（跨层或同层远距）"""
    # 类型校验确保conn是Connection实例
    if not isinstance(conn, Connection):
        raise TypeError("参数必须是Connection类型")

    # 判断跨层连接（Input到LIFNodes）
    if isinstance(conn.source, Input) and isinstance(conn.target, LIFNodes):
        return True  # 输入层到兴奋层的顶突触
    
    # 判断其他跨层连接（如LIFNodes到其他类型层）
    elif isinstance(conn.source, LIFNodes) and not isinstance(conn.target, LIFNodes):
        return True
    
    # 同层LIFNodes连接的判断
    if isinstance(conn.source, LIFNodes) and conn.source == conn.target:
        n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
        src_idx = conn.source_index
        tar_idx = conn.target_index
        src_pos = (src_idx//n_sqrt, src_idx%n_sqrt)
        tar_pos = (tar_idx//n_sqrt, tar_idx%n_sqrt)
        distance = np.sqrt((src_pos[0]-tar_pos[0])**2 + (src_pos[1]-tar_pos[1])**2)
        return distance > 5  # 空间距离阈值
        
    return False

# 标记所有连接的突触类型
for conn_name in network.connections:
    conn = network.connections[conn_name]
    conn.synapse_type = 'apical' if is_apical_connection(conn) else 'basal'

spike_history = torch.zeros(n_neurons, device=device)  # 记录神经元激活历史

# Directs network to GPU
if gpu:
    network.to("cuda")

# Voltage recording for excitatory and inhibitory layers.
#兴奋层和抑制层添加电压监视器
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
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

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
for i, datum in enumerate(dataloader):
    if i > n_train:
        break

    image = datum["encoded_image"]
    label = datum["label"]

    if i % update_interval == 0 and i > 0:
        # Get network predictions.
        all_activity_pred = all_activity(spike_record, assignments, n_classes)
        proportion_pred = proportion_weighting(spike_record, assignments, proportions, n_classes)

        # Compute network accuracy according to available classification strategies.
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

        # Assign labels to excitatory layer neurons.
        assignments, proportions, rates = assign_labels(
            spike_record, labels, n_classes, rates
        )

    # Add the current label to the list of labels for this update_interval
    labels[i % update_interval] = label[0]

    # Run the network on the input.
    choice = np.random.choice(int(n_neurons / n_classes), size=n_clamp, replace=False)
    clamp = {"Ae": per_class * label.long() + torch.Tensor(choice).long()}
    if gpu:
        inputs = {"X": image.cuda().view(time, 1, 1, 28, 28)}
    else:
        inputs = {"X": image.view(time, 1, 1, 28, 28)}
    network.run(inputs=inputs, time=time, clamp=clamp)

    # Get voltage recording.
    exc_voltages = exc_voltage_monitor.get("v")
    inh_voltages = inh_voltage_monitor.get("v")

    # Add to spikes recording.
    spike_record[i % update_interval] = spikes["Ae"].get("s").view(time, n_neurons)

    current_spikes = spikes["Ae"].get("s").sum(dim=0).squeeze()

    if i == 0:  # 预计算输入层邻近掩膜
        input_mask = get_neighborhood_mask((28,28), args.apical_radius).to(device)

        # === 权重更新部分 ===
    current_spikes = spikes["Ae"].get("s").sum(dim=0).squeeze()
    
    for conn_name in network.connections:
        conn = network.connections[conn_name]
        src_name = get_layer_name(network, conn.source)
        tar_name = get_layer_name(network, conn.target)
        if conn.synapse_type == 'apical':
            # 使用spike_history计算空间共活性
            if 'Ae' in conn.target.name: 
                spike_map = spike_history.view(n_sqrt, n_sqrt).T
                # 计算局部共活性（使用卷积核）
                co_activity = torch.conv2d(
                    spike_map.unsqueeze(0).unsqueeze(0),
                    torch.ones(1,1,5,5),  # 5x5邻域
                    padding=2
                ).squeeze()
                
                # 动态学习率
                lr = args.apical_lr[0] if i < args.early_phase else args.apical_lr[1]
                conn.w += lr * (co_activity.view(-1) - 0.6)  # 阈值0.6
            
        elif conn.synapse_type == 'basal':
            pass
            
    # 更新脉冲历史（指数衰减平均）
    spike_history = 0.85 * spike_history + 0.15 * current_spikes.float()

    # Optionally plot various simulation information.
    if plot:
        inpt = inputs["X"].view(time, 784).sum(0).view(28, 28)
        input_exc_weights = network.connections[("X", "Ae")].w
        square_weights = get_square_weights(
            input_exc_weights.view(784, n_neurons), n_sqrt, 28
        )
        square_assignments = get_square_assignments(assignments, n_sqrt)
        voltages = {"Ae": exc_voltages, "Ai": inh_voltages}

        inpt_axes, inpt_ims = plot_input(
            image.sum(1).view(28, 28), inpt, label=label, axes=inpt_axes, ims=inpt_ims
        )
        spike_ims, spike_axes = plot_spikes(
            {layer: spikes[layer].get("s").view(time, 1, -1) for layer in spikes},
            ims=spike_ims,
            axes=spike_axes,
        )

        # 修改后的绘图代码
        if conn.synapse_type == 'apical':
            plt.figure(figsize=(8,6))
            
            # 根据连接类型选择坐标
            if "X_to_Ae" in conn_name:  # 输入层到Ae的顶突触
                x = input_positions[:,1]  # 列坐标
                y = 28 - input_positions[:,0]  # 行坐标（翻转Y轴方向）
                c = conn.w.mean(1).cpu().numpy()  # 输入层每个位置到Ae的平均权重
            else:  # Ae层内部或其他顶突触
                x = positions[:,1]  # 列坐标
                y = n_sqrt - positions[:,0]  # 行坐标（翻转Y轴方向）
                c = conn.w.mean(0).cpu().numpy()  # Ae到目标层的平均权重
            
            plt.scatter(x, y, c=c, cmap='hot', s=50, edgecolor='k', linewidth=0.5)
            plt.colorbar(label='Mean Weight')
            plt.title(f"Spatial Distribution of {conn_name} Weights")
            plt.xticks([])
            plt.yticks([])
            plt.grid(alpha=0.3)

        weights_im = plot_weights(square_weights, im=weights_im)
        assigns_im = plot_assignments(square_assignments, im=assigns_im)
        perf_ax = plot_performance(accuracy, x_scale=update_interval, ax=perf_ax)
        #voltage_ims, voltage_axes = plot_voltages(
         #   voltages, ims=voltage_ims, axes=voltage_axes
        #)

        plt.pause(1e-8)

    network.reset_state_variables()  # Reset state variables.
    pbar.set_description_str("Train progress: ")
    pbar.update()

print("Progress: %d / %d \n" % (n_train, n_train))
print("Training complete.\n")

# 训练结束处添加：
torch.save({
    'network': network.state_dict(),
    'assignments': assignments,
    'proportions': proportions
}, "model.pth")

# 测试开始处添加：
checkpoint = torch.load("model.pth")
network.load_state_dict(checkpoint['network'])
print("Testing....\n")

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

pbar = tqdm(total=n_test)
for step, batch in enumerate(test_dataset):
    if step > n_test:
        break
    # Get next input sample.
    inputs = {"X": batch["encoded_image"].view(int(time / dt), 1, 1, 28, 28)}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    network.run(inputs=inputs, time=time)

    # Add to spikes recording.
    spike_record[0] = spikes["Ae"].get("s").squeeze()

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

    pbar.set_description_str(
        f"Accuracy: {(max(accuracy['all'] ,accuracy['proportion'] ) / (step+1)):.3}"
    )
    pbar.update()

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))

print("Testing complete.\n")
