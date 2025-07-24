"""
脉冲神经网络(SNN)工具模块
包含自定义学习规则、绘图功能、权重加载和保存功能
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from bindsnet.learning import LearningRule
from bindsnet.network.monitors import Monitor

# 设置matplotlib中文支持
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


class SpatialLearningRule(LearningRule):
    """
    空间邻域增强的STDP学习规则
    将传统STDP与空间邻域增强相结合，增强对特征的提取能力
    """
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
                                            delta_w[idx, j] += self.nu[0] * 0.01 / len(neighbor_region_coords)

        # 权重更新
        w += delta_w
        w.clamp_(wmin, wmax)
        self.connection.w.data.copy_(w)


def visualize_784x100_weights(weights, n_sqrt):
    """
    将784×100的权重矩阵可视化为一个网格，每个网格是28×28的图像
    
    参数:
        weights: 权重矩阵，形状为(784, 100)
        n_sqrt: 输出神经元的网格大小（通常为10，即10×10=100个神经元）
        
    返回:
        grid: 可视化后的网格图像
    """
    # 确保权重形状正确
    if weights.shape != (784, 100):
        if weights.shape == (100, 784):
            weights = weights.t()  # 转置
        else:
            reshaped = weights.view(784, -1)
            if reshaped.shape[1] >= 100:
                weights = reshaped[:, :100]
            else:
                raise ValueError(f"无法将权重调整为784×100，当前形状: {weights.shape}")
    
    # 确保权重在CPU上
    if weights.device.type != 'cpu':
        weights = weights.cpu()
    
    # 创建10×10的网格，每个格子是28×28的图像
    grid = torch.zeros((28 * n_sqrt, 28 * n_sqrt))
    
    for i in range(n_sqrt):
        for j in range(n_sqrt):
            neuron_idx = i * n_sqrt + j  # 神经元索引
            if neuron_idx < 100:
                neuron_weights = weights[:, neuron_idx].view(28, 28)  # 该神经元的权重
                
                # 填充到网格中的对应位置
                grid[i*28:(i+1)*28, j*28:(j+1)*28] = neuron_weights
    
    return grid.numpy()


def load_pretrained_weights(network, weights_path=None, metadata_path=None, device="cpu"):
    """
    加载预训练权重到网络中
    
    参数:
        network (Network): 要加载权重的网络
        weights_path (str): 权重文件路径
        metadata_path (str): 元数据文件路径
        device (str): 设备类型（"cpu"或"cuda"）
    
    返回:
        bool: 是否成功加载权重
    """
    if weights_path is None:
        weights_path = os.path.join("..", "..", "weights", "spatial_network_weights.pt")
    
    if metadata_path is None:
        metadata_path = os.path.join("..", "..", "weights", "spatial_network_metadata.pt")
    
    if os.path.exists(weights_path):
        try:
            weights = torch.load(weights_path, map_location=device)
            network.connections[("X", "E")].w = weights
            print(f"成功从 {weights_path} 加载权重")
            
            if os.path.exists(metadata_path):
                metadata = torch.load(metadata_path, map_location=device)
                print(f"加载权重元数据：{metadata}")
            
            return True
        except Exception as e:
            print(f"加载权重时出错: {e}")
    
    return False


def save_weights_evolution(weights, current_samples, n_sqrt, n_neurons, file_name, save_individual=True):
    """
    保存权重演变图
    
    参数:
        weights: 当前权重矩阵
        current_samples: 当前样本数
        n_sqrt: 输出神经元网格大小的平方根
        n_neurons: 神经元数量
        file_name: 脚本文件名(不含扩展名)，用于创建文件夹
        save_individual: 是否保存单个神经元的权重图
    """
    # 创建以Python文件名命名的文件夹
    weights_evolution_dir = os.path.join("..", "..", "weights_evolution", file_name)
    os.makedirs(weights_evolution_dir, exist_ok=True)
    
    # 创建当前样本数的子文件夹
    sample_dir = os.path.join(weights_evolution_dir, f"sample_{current_samples}")
    os.makedirs(sample_dir, exist_ok=True)
    
    print(f"\n正在保存权重到 {sample_dir}")
    
    # 确保输入权重格式正确
    if weights.shape != (784, n_neurons):
        weights = weights.view(784, n_neurons)
    
    # 如果权重在GPU上，先移到CPU
    if weights.device.type != 'cpu':
        weights = weights.cpu()
    
    # 生成总体权重网格
    weight_grid = visualize_784x100_weights(weights, n_sqrt)
    
    # 保存总体权重网格图
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(weight_grid, cmap="hot_r", vmin=0, vmax=1)
    ax.set_title(f"权重可视化 (784×{n_neurons}) - 样本 {current_samples}")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel(f"{n_sqrt}×{n_sqrt}神经元的输入权重矩阵")
    ax.set_ylabel(f"每个小方块是一个神经元的28×28输入权重")
    
    grid_path = os.path.join(sample_dir, "weights_grid.png")
    fig.savefig(grid_path, dpi=200)
    plt.close(fig)
    
    # 创建神经元网格图（每个神经元一个小格子）
    grid_size = n_sqrt
    n_neurons_to_show = min(grid_size * grid_size, n_neurons)
    
    grid_fig, grid_axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # 将axes转为一维数组
    grid_axes = grid_axes.flatten()
    
    # 准备颜色映射
    cmap = plt.cm.hot_r
    vmin, vmax = 0.0, 1.0
    
    # 按行填充神经元网格图
    for i, neuron_idx in enumerate(range(n_neurons_to_show)):
        if i < len(grid_axes):
            neuron_weights = weights[:, neuron_idx].view(28, 28).numpy()
            im = grid_axes[i].imshow(neuron_weights, cmap=cmap, vmin=vmin, vmax=vmax)
            grid_axes[i].set_xticks([])
            grid_axes[i].set_yticks([])
            grid_axes[i].set_frame_on(False)
    
    plt.tight_layout(pad=0.4)
    neurons_grid_path = os.path.join(sample_dir, "neurons_grid.png")
    grid_fig.savefig(neurons_grid_path, dpi=300, bbox_inches='tight')
    plt.close(grid_fig)
    
    # 可选地保存单个神经元的权重图
    if save_individual:
        neurons_dir = os.path.join(sample_dir, "neurons")
        os.makedirs(neurons_dir, exist_ok=True)
        
        for neuron_idx in range(n_neurons):
            neuron_weights = weights[:, neuron_idx].view(28, 28).numpy()
            plt.figure(figsize=(3, 3))
            plt.imshow(neuron_weights, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(neurons_dir, f"neuron_{neuron_idx}.png"), dpi=100)
            plt.close()
    
    print(f"已保存权重演变图到 {sample_dir}")


def save_final_weights(weights_history, current_samples, n_neurons, n_sqrt, network, input_exc_weights):
    """
    保存最终权重和元数据
    
    参数:
        weights_history: 权重历史记录
        current_samples: 当前样本数
        n_neurons: 神经元数量
        n_sqrt: 输出神经元网格大小的平方根
        network: 神经网络对象
        input_exc_weights: 最终权重
    """
    print("正在保存训练权重...")
    
    # 获取最终权重
    final_weights = weights_history[-1]['weights']
    
    # 创建保存目录
    weights_dir = os.path.join("..", "..", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    # 保存权重到本地
    weights_path = os.path.join(weights_dir, "spatial_network_weights.pt")
    torch.save(final_weights, weights_path)
    print(f"权重已保存至 {weights_path}")
    
    # 保存整个网络状态
    network_path = os.path.join(weights_dir, "spatial_network_full.pt")
    torch.save(network.state_dict(), network_path)
    print(f"网络状态已保存至 {network_path}")
    
    # 额外保存一些关键信息，便于以后加载
    metadata = {
        'n_neurons': n_neurons,
        'n_input': 784,
        'n_sqrt': n_sqrt,
        'trained_samples': current_samples,
        'learning_rule': 'SpatialLearningRule',
        'date': torch.datetime.now().strftime("%Y-%m-%d %H:%M:%S") if hasattr(torch, 'datetime') else None
    }
    
    metadata_path = os.path.join(weights_dir, "spatial_network_metadata.pt")
    torch.save(metadata, metadata_path)
    print(f"网络元数据已保存至 {metadata_path}")
    
    # 添加说明，表明权重图已经在训练过程中保存
    print("所有权重演变图已在训练过程中每隔 update_interval 个样本保存")
    
    # 检查最后一个样本文件夹，如果不存在则保存
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    weights_evolution_dir = os.path.join("..", "..", "weights_evolution", file_name)
    last_sample_dir = os.path.join(weights_evolution_dir, f"sample_{current_samples}")
    
    # 如果最后一个样本没有保存（可能因为不是update_interval的倍数），则额外保存
    if not os.path.exists(last_sample_dir):
        os.makedirs(last_sample_dir, exist_ok=True)
        print(f"\n正在保存最终权重到 {last_sample_dir}")
        
        # 确保输入权重格式正确并在CPU上
        if input_exc_weights.shape != (784, n_neurons):
            input_exc_weights = input_exc_weights.view(784, n_neurons)
        
        # 保存最终权重网格图
        save_weights_evolution(
            weights=input_exc_weights,
            current_samples=current_samples,
            n_sqrt=n_sqrt,
            n_neurons=n_neurons,
            file_name=file_name,
            save_individual=True
        )
    
    # 添加颜色条参考图到weights_evolution目录
    fig, ax = plt.subplots(figsize=(6, 1))
    cb = plt.colorbar(plt.cm.ScalarMappable(cmap="hot_r", norm=plt.Normalize(vmin=0.0, vmax=1.0)), 
                       cax=ax, orientation='horizontal')
    cb.set_label('权重值范围参考')
    colorbar_path = os.path.join(weights_evolution_dir, "colorbar_reference.png")
    plt.savefig(colorbar_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"权重演变已保存至 {weights_evolution_dir}")


def setup_monitors(network, time, device):
    """
    为网络设置监视器
    
    参数:
        network: 神经网络对象
        time: 时间步长
        device: 设备类型
    
    返回:
        spikes: 脉冲监视器
        voltages: 电压监视器
        weight_monitor: 权重监视器
    """
    # 添加脉冲监视器
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
    
    return spikes, voltages, weight_monitor 