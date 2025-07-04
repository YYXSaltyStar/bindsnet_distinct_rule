from typing import Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from scipy.spatial.distance import euclidean
from torch.nn.modules.utils import _pair

from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import DiehlAndCookNodes, Input, LIFNodes
from bindsnet.network.topology import Connection, LocalConnection
from bindsnet.learning import LearningRule

class SpatialLearningRule(LearningRule):
    def __init__(self, connection, nu, **kwargs):
        super().__init__(connection=connection, nu=nu)
        self.device = connection.source.s.device
        self.input_shape = kwargs.get('input_shape', (28, 28))
        self.output_shape = kwargs.get('output_shape', (10, 10))
        self.threshold = kwargs.get('threshold', 0.15)
        self.window = kwargs.get('window', 10)
        self.neighbor_radius = kwargs.get('neighbor_radius', 1)#对于3*3的区域
        self.tau = kwargs.get('tau', 1.0)

    def convert_input_to_E(self, pre_y, pre_x):
        """给出输入层的一个神经元的坐标,返回输出层的一个神经元的坐标"""
        scale_y = self.output_shape[0] / self.input_shape[0]#10/28
        scale_x = self.output_shape[1] / self.input_shape[1]#10/28
        post_y = int(pre_y * scale_y)#10/28*y
        post_x = int(pre_x * scale_x)#10/28*x
        return post_y, post_x

    def get_input_coordinates(self, pre_y, pre_x):
        """获取输出层上一个神经元对应的输入层区域矩阵"""
        scale_y = self.input_shape[0] / self.output_shape[0]#28/10
        scale_x = self.input_shape[1] / self.output_shape[1]#28/10
        
        # 计算区域范围（对于输出层的一个单点）
        start_y = int(pre_y * scale_y)#28/10*y
        start_x = int(pre_x * scale_x)#28/10*x
        end_y = int((pre_y + 1) * scale_y)#28/10*(y+1)
        end_x = int((pre_x + 1) * scale_x)#28/10*(x+1)
        
        # 生成区域内的所有坐标
        coords = []
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                if 0 <= y < self.input_shape[0] and 0 <= x < self.input_shape[1]:
                    coords.append((y, x))
        return coords

    def get_region_spikes(self, pre_spikes, region_coords, t, window):
        """输入之前获得的region_coords,返回区域内的脉冲活动"""
        region_spikes = []
        """用来填入之前input层神经元矩阵的脉冲情况
        假设有四个神经元,时间窗口为4,则regin_spokes长这样:
        [[1,0,0,0],
        [0,1,0,0],
        [1,0,1,0],
        [0,1,0,0]]
        """
        for y, x in region_coords:
            #先把二维的坐标(x,y)转换为一维的索引,这是为了配合监视器上的索引
            idx = y * self.input_shape[1] + x
            spikes = pre_spikes[t:t+window, idx]#得到一个时间窗口上的脉冲[1,0,0,0]
            region_spikes.append(spikes)
        return torch.stack(region_spikes).max(dim=0)[0]  # 取最大值
        """最后会得到一个一维的向量,长度为4,表示这个区域在时间窗口内的脉冲情况[1,1,1,0]"""

    def update(self, **kwargs):
        pre_spikes = self.connection.source.s.float().to(self.device)
        post_spikes = self.connection.target.s.float().to(self.device)
        
        # 重塑张量维度
        if len(pre_spikes.shape) == 4:  # 例如, 输入层脉冲可能是 [batch_size, channels, height, width] -> [1, 1, 28, 28]
            pre_spikes = pre_spikes.view(pre_spikes.shape[0], pre_spikes.shape[1], -1)  # 重塑为 [1, 1, 28*28]
        if len(post_spikes.shape) == 4:
            post_spikes = post_spikes.view(1, 1, -1)
        
        # 检查并调整维度
        if len(pre_spikes.shape) == 2:  # 如果只有两个维度 (T, N)
            pre_spikes = pre_spikes.unsqueeze(1)  # 添加批次维度 (T, 1, N)
        if len(post_spikes.shape) == 2:
            post_spikes = post_spikes.unsqueeze(1)
        
        T, B, N_pre = pre_spikes.shape
        _, _, N_post = post_spikes.shape
        pre_spikes = pre_spikes[:, 0, :]# pre_spikes从[250, 1, 784]变为[250, 784]
        post_spikes = post_spikes[:, 0, :]# post_spikes从[250, 1, 100]变为[250,1, 100]

        # 确保所有权重相关张量在正确的设备上
        w = self.connection.w.data.to(self.device)
        wmin = torch.tensor(self.connection.wmin, device=self.device)
        wmax = torch.tensor(self.connection.wmax, device=self.device)
        delta_w = torch.zeros_like(w, device=self.device)
        
        # 遍历所有时间步,逐时间处理STDP和空间邻域增强
        for t in range(T - self.window):  # 确保有足够的窗口大小
            # 遍历所有连接
            for i in range(N_pre):#784次循环
                for j in range(N_post):#100次循环
                    # 获取时间窗口内的脉冲
                    pre_window = pre_spikes[t:t+self.window, i]#例如[0,1,0,0,0,0,0,0,0,0]表示在窗口的第2个时间步有脉冲
                    post_window = post_spikes[t:t+self.window, j]
                    
                    # STDP学习
                    """pre_times为tensor([1]),表示脉冲发生在索引1处
                    post_times为tensor([3,8]),表示脉冲发生在索引3和8处"""
                    pre_times = (pre_window > 0).nonzero(as_tuple=True)[0]
                    post_times = (post_window > 0).nonzero(as_tuple=True)[0]
                    
                    if len(pre_times) > 0 and len(post_times) > 0:
                        # 计算所有可能的时间差,例如上面距离的，分别为3-1和8-1
                        time_diffs = post_times.unsqueeze(0) - pre_times.unsqueeze(1)#unsqueeze(0)和unsqueeze(1)是为了将tensor变成一个纵向向量和一个横向向量
                        # 只保留前神经元先发放的情况
                        valid_diffs = time_diffs[time_diffs > 0]
                        
                        if len(valid_diffs) > 0:
                            # 计算权重更新量
                            updates = self.nu[0] * torch.exp(-valid_diffs / self.tau)
                            # 累加所有有效的更新量
                            delta_w[i, j] += updates.sum()
                    
                    # 空间邻域增强 - 保留空间映射功能
                    pre_y, pre_x = divmod(i, self.input_shape[1])#divmod同时返回商和余数，可以直接将一维的输入层索引i转换为二维的坐标(x,y)
                    for dy in range(-self.neighbor_radius, self.neighbor_radius+1):
                        for dx in range(-self.neighbor_radius, self.neighbor_radius+1):
                            ni, nj = pre_y+dy, pre_x+dx# 在输入层中的神经元
                            if 0 <= ni < self.input_shape[0] and 0 <= nj < self.input_shape[1]:
                                n_idx = ni * self.input_shape[1] + nj# 将二维坐标转换为一维索引，仅仅是为了能够使用for循环，并保证不超过输入层神经元数量
                                if n_idx < N_pre:
                                    """这里的思路有点绕。我们有了 输入层 的一个小矩阵中的一个神经元的二维索引(x,y)
                                    我们想得到这个神经元在 输入层 中的所有邻居的二维索引，
                                    那就先转换成 输出层 的一个神经元的二维索引，
                                    然后通过函数get_input_coordinates，得到这个神经元在 输入层 中的所有邻居的二维索引
                                    然后再通过函数get_region_spikes，得到这个神经元在 输入层 中的所有邻居的脉冲情况
                                    """
                                    neighbor_region_coords = self.get_input_coordinates(#得到了输入层的矩阵
                                        self.convert_input_to_E(ni, nj)[0],
                                        self.convert_input_to_E(ni, nj)[1]
                                    )
                                    neighbor_pre_spikes = self.get_region_spikes(
                                        pre_spikes, neighbor_region_coords, t, self.window
                                    )
                                    
                                    if len((neighbor_pre_spikes > 0).nonzero(as_tuple=True)[0]) > 0 and \
                                       len(post_times) > 0:
                                        """第一行：判断输入层上，时间窗口中，是否有神经元活动
                                        第二行：判断输出层上，时间窗口中，是否有神经元活动
                                        0.5/矩阵面积(为了平摊)倍率的nu(试着调参)
                                        """
                                        for y, x in neighbor_region_coords:
                                            idx = y * self.input_shape[1] + x
                                            #idx指的是在输入层，即感受野上的神经元矩阵；j表示一个输出层的神经元
                                            delta_w[idx, j] += self.nu[0] * 0.5 / len(neighbor_region_coords)#这里除以len(neighbor_region_coords)是为了将权重按照空间平均分配到邻居神经元上

        # 权重更新
        w += delta_w
        w.clamp_(wmin, wmax)
        self.connection.w.data.copy_(w)


class DiehlAndCook2015_with_SPL(Network):
    # language=rst
    """
    Implements the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_.
    """

    def __init__(
        self,
        n_inpt: int,
        n_neurons: int = 100,
        exc: float = 22.5,
        inh: float = 17.5,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        inpt_shape: Optional[Iterable[int]] = None,
        inh_thresh: float = -40.0,
        exc_thresh: float = -52.0,
    ) -> None:
        # language=rst
        """
        Constructor for class ``DiehlAndCook2015``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param exc: Strength of synapse weights from excitatory to inhibitory layer.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons = n_neurons
        self.exc = exc
        self.inh = inh
        self.dt = dt

        # Layers
        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        exc_layer = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=exc_thresh,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        inh_layer = LIFNodes(
            n=self.n_neurons,
            traces=False,
            rest=-60.0,
            reset=-45.0,
            thresh=inh_thresh,
            tc_decay=10.0,
            refrac=2,
            tc_trace=20.0,
        )

        # Connections
        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
        input_exc_conn = Connection(
            source=input_layer,
            target=exc_layer,
            w=w,
            update_rule=SpatialLearningRule,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        w = self.exc * torch.diag(torch.ones(self.n_neurons))
        exc_inh_conn = Connection(
            source=exc_layer, target=inh_layer, w=w, wmin=0, wmax=self.exc
        )
        w = -self.inh * (
            torch.ones(self.n_neurons, self.n_neurons)
            - torch.diag(torch.ones(self.n_neurons))
        )
        inh_exc_conn = Connection(
            source=inh_layer, target=exc_layer, w=w, wmin=-self.inh, wmax=0
        )

        # Add to network
        self.add_layer(input_layer, name="X")
        self.add_layer(exc_layer, name="Ae")
        self.add_layer(inh_layer, name="Ai")
        self.add_connection(input_exc_conn, source="X", target="Ae")
        self.add_connection(exc_inh_conn, source="Ae", target="Ai")
        self.add_connection(inh_exc_conn, source="Ai", target="Ae")
