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

from typing import Optional, Sequence, Union, Tuple
import torch
from torch.nn.modules.utils import _pair
import torch.nn.functional as F

from bindsnet.learning import PostPre, LearningRule
from bindsnet.network.topology import AbstractConnection, Connection


class CombinedSpatialPostPre(PostPre):
    """
    结合了标准STDP(PostPre)和空间邻域增强(SpatialLearningRule)的学习规则。
    通过继承PostPre类获取其高效向量化的STDP实现，同时整合空间邻域增强逻辑。
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,#nu之所以变成这样，是因为之前产生了许多张量不匹配、float不匹配的问题
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        """
        CombinedSpatialPostPre学习规则的构造函数。

        :param connection: 将被此学习规则修改权重的AbstractConnection对象。
        :param nu: 学习率。可以是：
              - 单个浮点数：将被用作所有学习率
              - 包含两个浮点数的列表/元组：用于STDP的前后突触更新
              - 包含三个浮点数的列表/元组：前两个用于STDP，第三个用于空间学习
        :param reduction: 用于在批次维度上减少参数更新的方法。
        :param weight_decay: 控制每次迭代权重衰减率的系数。
        :param kwargs: 空间学习所需的其他参数，如input_shape, output_shape等。
        """
        # nu参数的处理过程，避免出现张量问题
        if nu is None:
            self.nu_spatial = 1e-4  # 默认空间学习率
            nu_stdp = nu
        elif isinstance(nu, (list, tuple)) and len(nu) >= 3:
            self.nu_spatial = float(nu[2])  # 确保是浮点数
            nu_stdp = nu[:2]
        elif isinstance(nu, (list, tuple)) and len(nu) == 2:
            self.nu_spatial = 1e-4  # 默认空间学习率
            nu_stdp = nu
        else:
            # 如果是单个值（浮点数或张量），将其用于所有学习率
            self.nu_spatial = float(nu) if isinstance(nu, (int, float)) else 1e-4
            nu_stdp = nu


        # 调用父类构造函数，传递STDP相关参数
        super().__init__(
            connection=connection,
            nu=nu_stdp,
            reduction=reduction,
            weight_decay=weight_decay,
        )

        # 存储空间学习规则所需的参数
        self.input_shape = kwargs.get('input_shape')
        self.output_shape = kwargs.get('output_shape')
        self.window = kwargs.get('window', 10)
        self.neighbor_radius = kwargs.get('neighbor_radius', 1)
        self.tau = kwargs.get('tau', 1.0)
        # 确保设备正确设置
        if hasattr(connection, 'w') and connection.w is not None:
            self.device = connection.w.device
        elif hasattr(connection.source, 's') and connection.source.s is not None:
            self.device = connection.source.s.device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def convert_input_to_E(self, pre_y: int, pre_x: int) -> Tuple[int, int]:
        """
        将输入层坐标映射到输出层坐标。

        :param pre_y: 输入层的y坐标
        :param pre_x: 输入层的x坐标
        :return: 对应的输出层坐标 (post_y, post_x)
        """
        scale_y = self.output_shape[0] / self.input_shape[0]
        scale_x = self.output_shape[1] / self.input_shape[1]
        post_y = int(pre_y * scale_y)
        post_x = int(pre_x * scale_x)
        return post_y, post_x

    def get_input_coordinates(self, pre_y: int, pre_x: int) -> list:
        """
        获取输出层上一个神经元对应的输入层区域矩阵坐标。

        :param pre_y: 输出层的y坐标
        :param pre_x: 输出层的x坐标
        :return: 输入层区域内的所有坐标列表
        """
        scale_y = self.input_shape[0] / self.output_shape[0]
        scale_x = self.input_shape[1] / self.output_shape[1]
        
        start_y = int(pre_y * scale_y)
        start_x = int(pre_x * scale_x)
        end_y = int((pre_y + 1) * scale_y)
        end_x = int((pre_x + 1) * scale_x)
        
        coords = []
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                if 0 <= y < self.input_shape[0] and 0 <= x < self.input_shape[1]:
                    coords.append((y, x))
        return coords

    def get_region_spikes(self, pre_spikes: torch.Tensor, region_coords: list, t: int, window: int) -> torch.Tensor:
        """
        获取区域内的脉冲活动。

        :param pre_spikes: 输入层的脉冲张量
        :param region_coords: 区域坐标列表
        :param t: 当前时间步
        :param window: 时间窗口大小
        :return: 区域内的最大脉冲活动
        """
        region_spikes = []
        for y, x in region_coords:
            idx = y * self.input_shape[1] + x
            spikes = pre_spikes[t:t+window, idx]
            region_spikes.append(spikes)
        return torch.stack(region_spikes).max(dim=0)[0]  # 取最大值

    def _connection_update(self, **kwargs) -> None:
        """
        重写Connection类型的更新方法，结合STDP和空间邻域增强。
        
        此方法分三步进行：
        1. 计算PostPre(STDP)的权重更新量
        2. 计算空间邻域增强的权重更新量
        3. 将两种更新量相加并应用到连接权重上
        """
        batch_size = self.source.batch_size
        
        # 初始化权重更新量
        delta_w_postpre = torch.zeros_like(self.connection.w, device=self.device)
        
        # 步骤A: 计算PostPre(STDP)的权重更新量
        # 这部分代码从PostPre._connection_update方法复制而来，
        # 但修改为累积更新量而不是直接应用到权重上
        
        # 前突触更新 (LTD: Long-Term Depression)
        if self.nu[0].any():
            source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float().to(self.device)
            target_x = (self.target.x.view(batch_size, -1).unsqueeze(1) * self.nu[0]).to(self.device)
            delta_w_postpre -= self.reduction(torch.bmm(source_s, target_x), dim=0)
            del source_s, target_x

        # 后突触更新 (LTP: Long-Term Potentiation)
        if self.nu[1].any():
            target_s = (
                self.target.s.view(batch_size, -1).unsqueeze(1).float() * self.nu[1]
            ).to(self.device)
            source_x = self.source.x.view(batch_size, -1).unsqueeze(2).to(self.device)
            delta_w_postpre += self.reduction(torch.bmm(source_x, target_s), dim=0)
            del source_x, target_s
        
        #步骤B: 计算空间邻域增强的权重更新量
        delta_w_spatial = torch.zeros_like(self.connection.w, device=self.device)
        
        # 获取并处理脉冲数据
        pre_spikes = self.connection.source.s.float().to(self.device)
        post_spikes = self.connection.target.s.float().to(self.device)
        
        # 维度处理
        if len(pre_spikes.shape) == 4:  # 例如 [batch_size, channels, height, width]
            pre_spikes = pre_spikes.view(pre_spikes.shape[0], pre_spikes.shape[1], -1)
        if len(post_spikes.shape) == 4:
            post_spikes = post_spikes.view(1, 1, -1)
        
        if len(pre_spikes.shape) == 2:  # 如果只有两个维度 (T, N)
            pre_spikes = pre_spikes.unsqueeze(1)  # 添加批次维度 (T, 1, N)
        if len(post_spikes.shape) == 2:
            post_spikes = post_spikes.unsqueeze(1)
        
        T, B, N_pre = pre_spikes.shape
        _, _, N_post = post_spikes.shape
        pre_spikes = pre_spikes[:, 0, :]  # 从[T, B, N]变为[T, N]
        post_spikes = post_spikes[:, 0, :]
        
        # 实现空间邻域增强逻辑
        for t in range(T - self.window):  # 确保有足够的窗口大小
            for i in range(N_pre):
                pre_y, pre_x = divmod(i, self.input_shape[1])
                for dy in range(-self.neighbor_radius, self.neighbor_radius+1):
                    for dx in range(-self.neighbor_radius, self.neighbor_radius+1):
                        ni, nj = pre_y+dy, pre_x+dx
                        if 0 <= ni < self.input_shape[0] and 0 <= nj < self.input_shape[1]:
                            n_idx = ni * self.input_shape[1] + nj
                            if n_idx < N_pre:
                                neighbor_region_coords = self.get_input_coordinates(
                                    *self.convert_input_to_E(ni, nj)
                                )
                                neighbor_pre_spikes = self.get_region_spikes(
                                    pre_spikes, neighbor_region_coords, t, self.window
                                )
                                post_times = (post_spikes[t:t+self.window, :] > 0).nonzero(as_tuple=True)[0]
                                if len((neighbor_pre_spikes > 0).nonzero(as_tuple=True)[0]) > 0 and len(post_times) > 0:
                                    for j in range(N_post):
                                        # 使用专门的空间学习率
                                        delta_w_spatial[n_idx, j] += self.nu_spatial * 0.5 / len(neighbor_region_coords)
        
        #步骤C: 组合并应用更新
        # 将两种更新量相加并应用到连接权重上
        total_delta_w = delta_w_postpre + delta_w_spatial
        # 确保权重和更新量在同一设备上
        if self.connection.w.device != total_delta_w.device:
            total_delta_w = total_delta_w.to(self.connection.w.device)
        self.connection.w += total_delta_w
        
        #步骤D: 处理权重衰减
        # 调用父类的update方法来处理权重衰减
        LearningRule.update(self)  #


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
            update_rule=CombinedSpatialPostPre,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )

        input_exc_conn = Connection(
            source=input_layer,
            target=exc_layer,
            w=w,
            update_rule=PostPre,
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


