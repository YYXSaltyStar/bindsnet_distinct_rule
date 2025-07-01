import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from bindsnet.network.topology import Connection
from bindsnet.analysis.plotting import plot_weights
from bindsnet.datasets import FashionMNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights
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
        scale_y = self.output_shape[0] / self.input_shape[0] #10/28,0.357左右
        scale_x = self.output_shape[1] / self.input_shape[1] #10/28,0.357左右
        post_y = int(pre_y * scale_y)#比如输入层的0，1，2都会对应到输出层的0
        post_x = int(pre_x * scale_x)
        return post_y, post_x

    def get_region_coordinates(self, pre_y, pre_x):
        """获取输出层中单个神经元，对应输入层中多个神经元的区域坐标"""
        scale_y = self.input_shape[0] / self.output_shape[0] #28/10,2.8
        scale_x = self.input_shape[1] / self.output_shape[1] #28/10,2.8
        
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
            region_spikes.append(spikes)#一个列表，每一个元素都是一个时间窗口内的脉冲活动
        return torch.stack(region_spikes).max(dim=0)[0]  # 堆叠成二维张量，，每一个时间步上只要有一个神经元发放就算发放

    def update(self, **kwargs):
        pre_spikes = self.connection.source.s.float().to(self.device)#获取输入层的脉冲活动
        post_spikes = self.connection.target.s.float().to(self.device)#获取兴奋层的脉冲活动
        
        # 重塑张量维度
        if len(pre_spikes.shape) == 4:  # 如果是 [1, 1, 28, 28]
            pre_spikes = pre_spikes.view(1, 1, -1)  # 重塑为 [1, 1, 784]
        if len(post_spikes.shape) == 4:
            post_spikes = post_spikes.view(1, 1, -1)
        
        # 检查并调整维度，要不然network.run()时老是报错
        if len(pre_spikes.shape) == 2:  # 如果只有两个维度 (T, N)
            pre_spikes = pre_spikes.unsqueeze(1)  # 添加批次维度 (T, 1, N)
        if len(post_spikes.shape) == 2:
            post_spikes = post_spikes.unsqueeze(1)
        
        T, B, N_pre = pre_spikes.shape
        _, _, N_post = post_spikes.shape
        pre_spikes = pre_spikes[:, 0, :]#把batchsize这一个维度先去掉，方便后续正向传播
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
                    pre_times = (pre_window > 0).nonzero(as_tuple=True)[0]#比如：1，5，7
                    post_times = (post_window > 0).nonzero(as_tuple=True)[0]#比如：2，6，9
                    
                    if len(pre_times) > 0 and len(post_times) > 0:
                        # 计算所有可能的时间差
                        time_diffs = post_times.unsqueeze(0) - pre_times.unsqueeze(1)
                        # 只保留前神经元先发放的情况
                        valid_diffs = time_diffs[time_diffs > 0]#时间差越大，w调整越小
                        
                        if len(valid_diffs) > 0:
                            # 计算权重更新量
                            updates = self.nu[0] * torch.exp(-valid_diffs / self.tau)
                            # 累加所有有效的更新量
                            delta_w[i, j] += updates.sum()
                    
                    # 空间邻域增强 - 保留空间映射功能
                    pre_y, pre_x = divmod(i, self.input_shape[1])#转二维
                    for dy in range(-self.neighbor_radius, self.neighbor_radius+1):
                        for dx in range(-self.neighbor_radius, self.neighbor_radius+1):
                            ni, nj = pre_y+dy, pre_x+dx
                            if 0 <= ni < self.input_shape[0] and 0 <= nj < self.input_shape[1]:
                                n_idx = ni * self.input_shape[1] + nj#转一维，获取脉冲
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
parser.add_argument("--n_train", type=int, default=60000)
parser.add_argument("--n_test", type=int, default=10000)
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
n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity
per_class = int(n_neurons / n_classes)

# 创建网络对象
torch.manual_seed(seed)
network = Network(dt=dt)

# 输入层（784个神经元，适配28x28图像）
input_layer = Input(n=784, shape=(1, 28, 28), traces=True)
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
    w=0.3 * torch.rand(784, n_neurons),
    update_rule=SpatialLearningRule,
    nu=1e-2,
    wmin=0.0,
    wmax=1.0,
    input_shape=(28, 28),
    output_shape=(10, 10),
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
dataset = FashionMNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "FashionMNIST"),
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
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

# 添加电压监视器
voltages = {}
for layer in set(network.layers) - {"X"}:  # 不监视输入层的电压
    voltages[layer] = Monitor(network.layers[layer], state_vars=["v"], time=time, device=device)
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

# 添加权重监视器 - 特别监视输入层到兴奋层的连接权重
weight_monitor = Monitor(
    network.connections[("X", "E")],
    state_vars=["w"],
    time=time,
    device=device
)
network.add_monitor(weight_monitor, name="XE_weights")

# 训练网络
print("开始训练...\n")

# 初始化权重绘图变量
weights_fig = None
weights_im = None
weights_history = []  # 存储每个时间步的权重
sample_interval = 100  # 每100个样本记录一次权重，避免存储过多数据

# 计算实际的训练步数
n_steps = (n_train + batch_size - 1) // batch_size
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
        clamp = {"E": per_class * labels_batch[b].long() + torch.Tensor(choice).long()}
        
        if gpu:
            inputs = {"X": images[b].cuda().view(time, 1, 1, 28, 28)}
        else:
            inputs = {"X": images[b].view(time, 1, 1, 28, 28)}
            
        network.run(inputs=inputs, time=time, clamp=clamp)

        # 记录脉冲
        spike_record[current_samples % update_interval] = spikes["E"].get("s").view(time, n_neurons)

        # 可视化权重
        if plot and current_samples % sample_interval == 0:
            # 获取当前权重
            input_exc_weights = network.connections[("X", "E")].w
            
            # 保存权重历史
            weights_history.append({
                'sample': current_samples,
                'weights': input_exc_weights.clone().detach().cpu()
            })
            
            # 绘制当前权重
            square_weights = get_square_weights(
                input_exc_weights.view(784, n_neurons), n_sqrt, 28
            )
            
            if weights_fig is None:
                weights_fig, ax = plt.subplots(figsize=(12, 10))
                weights_im = ax.imshow(square_weights, cmap="hot_r", vmin=0, vmax=1)
                ax.set_title(f"权重可视化 - 样本 {current_samples}")
                plt.colorbar(weights_im, ax=ax)
            else:
                weights_im.set_data(square_weights)
                weights_fig.suptitle(f"权重可视化 - 样本 {current_samples}")
                
            # 保存当前权重图
            weight_img_path = os.path.join("..", "..", "plots", "weights", f"weights_{current_samples}.png")
            os.makedirs(os.path.dirname(weight_img_path), exist_ok=True)
            weights_fig.savefig(weight_img_path)
            
            plt.pause(1e-8)

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

# 保存权重历史动画
if plot and len(weights_history) > 0:
    print("正在生成权重演变动画...")
    
    # 创建一个更详细的权重演变图
    weights_animation_fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes = axes.flatten()
    
    # 初始化第一帧
    first_weights = weights_history[0]['weights']
    first_square = get_square_weights(first_weights.view(784, n_neurons), n_sqrt, 28)
    
    # 全局权重图
    weights_animation_im = axes[0].imshow(first_square, cmap="hot_r", vmin=0, vmax=1)
    axes[0].set_title("全局权重矩阵")
    plt.colorbar(weights_animation_im, ax=axes[0])
    
    # 选择几个特定的神经元进行跟踪
    tracked_neurons = [0, int(n_neurons/4), int(n_neurons/2), int(3*n_neurons/4)]
    neuron_weights_ims = []
    
    for i, neuron_idx in enumerate(tracked_neurons):
        if i+1 < len(axes):  # 确保我们有足够的子图
            neuron_weights = first_weights[:, neuron_idx].view(28, 28)
            im = axes[i+1].imshow(neuron_weights, cmap="hot_r", vmin=0, vmax=1)
            axes[i+1].set_title(f"神经元 #{neuron_idx} 的权重")
            plt.colorbar(im, ax=axes[i+1])
            neuron_weights_ims.append(im)
    
    # 创建一个图表来显示权重变化趋势
    weight_trends = []
    samples = [data['sample'] for data in weights_history]
    
    for neuron_idx in tracked_neurons:
        trend = [data['weights'][:, neuron_idx].mean().item() for data in weights_history]
        weight_trends.append(trend)
    
    def update_weights_animation(frame):
        if frame < len(weights_history):
            current_data = weights_history[frame]
            current_weights = current_data['weights']
            current_square = get_square_weights(current_weights.view(784, n_neurons), n_sqrt, 28)
            
            # 更新全局权重图
            weights_animation_im.set_data(current_square)
            axes[0].set_title(f"全局权重矩阵 - 样本 {current_data['sample']}")
            
            # 更新每个跟踪神经元的权重图
            for i, neuron_idx in enumerate(tracked_neurons):
                if i+1 < len(axes):
                    neuron_weights = current_weights[:, neuron_idx].view(28, 28)
                    neuron_weights_ims[i].set_data(neuron_weights)
                    axes[i+1].set_title(f"神经元 #{neuron_idx} 的权重 - 样本 {current_data['sample']}")
            
        return [weights_animation_im] + neuron_weights_ims
    
    from matplotlib.animation import FuncAnimation
    
    # 创建动画
    anim = FuncAnimation(
        weights_animation_fig, 
        update_weights_animation,
        frames=len(weights_history),
        interval=300,  # 每帧间隔时间(毫秒)
        blit=True
    )
    
    # 保存动画
    anim_path = os.path.join("..", "..", "plots", "weights", "weights_evolution.mp4")
    os.makedirs(os.path.dirname(anim_path), exist_ok=True)
    anim.save(anim_path, writer='ffmpeg', fps=3)
    print(f"权重演变动画已保存至 {anim_path}")
    
    # 绘制权重变化趋势图
    trend_fig, ax = plt.subplots(figsize=(12, 8))
    for i, neuron_idx in enumerate(tracked_neurons):
        ax.plot(samples, weight_trends[i], label=f"神经元 #{neuron_idx}")
    
    ax.set_xlabel("训练样本数")
    ax.set_ylabel("平均权重")
    ax.set_title("跟踪神经元的权重变化趋势")
    ax.legend()
    ax.grid(True)
    
    # 保存趋势图
    trend_path = os.path.join("..", "..", "plots", "weights", "weight_trends.png")
    trend_fig.savefig(trend_path)
    print(f"权重变化趋势图已保存至 {trend_path}")

print("开始测试...\n")

# 加载测试数据
test_dataset = FashionMNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "FashionMNIST"),
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
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

        # 可选地绘制权重
        if plot and current_samples % 10 == 0:  # 每10个样本绘制一次
            input_exc_weights = network.connections[("X", "E")].w
            
            # 获取当前输入
            if gpu:
                current_input = images[b].cuda().view(time, 784).sum(0).view(28, 28).cpu().numpy()
            else:
                current_input = images[b].view(time, 784).sum(0).view(28, 28).numpy()
            
            # 创建一个2x2的子图布局
            if weights_fig is None:
                weights_fig, axes = plt.subplots(2, 2, figsize=(16, 14))
                axes = axes.flatten()
                
                # 全局权重图
                square_weights = get_square_weights(
                    input_exc_weights.view(784, n_neurons), n_sqrt, 28
                )
                weights_im = axes[0].imshow(square_weights, cmap="hot_r", vmin=0, vmax=1)
                axes[0].set_title(f"全局权重矩阵 - 测试样本 {current_samples}")
                plt.colorbar(weights_im, ax=axes[0])
                
                # 当前输入图像
                input_im = axes[1].imshow(current_input, cmap="gray")
                axes[1].set_title(f"当前输入 - 标签: {labels_batch[b]}")
                plt.colorbar(input_im, ax=axes[1])
                
                # 选择两个神经元进行详细可视化
                neuron1_idx = int(n_neurons/3)
                neuron2_idx = int(2*n_neurons/3)
                
                neuron1_weights = input_exc_weights[:, neuron1_idx].view(28, 28)
                neuron1_im = axes[2].imshow(neuron1_weights, cmap="hot_r", vmin=0, vmax=1)
                axes[2].set_title(f"神经元 #{neuron1_idx} 的权重")
                plt.colorbar(neuron1_im, ax=axes[2])
                
                neuron2_weights = input_exc_weights[:, neuron2_idx].view(28, 28)
                neuron2_im = axes[3].imshow(neuron2_weights, cmap="hot_r", vmin=0, vmax=1)
                axes[3].set_title(f"神经元 #{neuron2_idx} 的权重")
                plt.colorbar(neuron2_im, ax=axes[3])
                
                # 保存当前权重图
                weight_img_path = os.path.join("..", "..", "plots", "weights", f"test_weights_{current_samples}.png")
                os.makedirs(os.path.dirname(weight_img_path), exist_ok=True)
                weights_fig.savefig(weight_img_path)
            else:
                axes = weights_fig.axes
                
                # 更新全局权重图
                square_weights = get_square_weights(
                    input_exc_weights.view(784, n_neurons), n_sqrt, 28
                )
                axes[0].images[0].set_data(square_weights)
                axes[0].set_title(f"全局权重矩阵 - 测试样本 {current_samples}")
                
                # 更新当前输入图像
                axes[1].images[0].set_data(current_input)
                axes[1].set_title(f"当前输入 - 标签: {labels_batch[b]}")
                
                # 更新神经元权重图
                neuron1_idx = int(n_neurons/3)
                neuron2_idx = int(2*n_neurons/3)
                
                neuron1_weights = input_exc_weights[:, neuron1_idx].view(28, 28)
                axes[2].images[0].set_data(neuron1_weights)
                
                neuron2_weights = input_exc_weights[:, neuron2_idx].view(28, 28)
                axes[3].images[0].set_data(neuron2_weights)
                
                # 保存更新后的权重图
                weight_img_path = os.path.join("..", "..", "plots", "weights", f"test_weights_{current_samples}.png")
                os.makedirs(os.path.dirname(weight_img_path), exist_ok=True)
                weights_fig.savefig(weight_img_path)
            
            plt.pause(1e-8)

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

# 如果有绘图窗口，保持它们打开
if plot:
    plt.show() 