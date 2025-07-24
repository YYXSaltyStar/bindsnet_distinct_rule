import argparse
import os
from time import time as t

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from bindsnet import ROOT_DIR
from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import DiehlAndCook2015
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights
from snn_utils import SpatialLearningRule
from bindsnet.learning import PostPre

def create_custom_network(args, device):
    """
    创建自定义的空间学习网络
    
    Args:
        args: 命令行参数
        device: 计算设备
    
    Returns:
        Network: 自定义网络实例
    """
    network = Network(dt=args.dt)
    
    # 输入层（784个神经元，适配28x28图像）
    input_layer = Input(n=784, shape=(1, 28, 28), traces=True)
    # 兴奋层
    exc_layer = LIFNodes(n=args.n_neurons, traces=True, theta_plus=args.theta_plus)
    # 抑制层
    inh_layer = LIFNodes(n=args.n_neurons, traces=True)
    
    # 添加层到网络
    network.add_layer(input_layer, name="X")
    network.add_layer(exc_layer, name="E")
    network.add_layer(inh_layer, name="I")
    
    # 输入层到兴奋层的连接（可学习）
    input_exc_conn = Connection(
        source=input_layer,
        target=exc_layer,
        w=0.3 * torch.rand(784, args.n_neurons),
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
        w=args.exc * torch.eye(args.n_neurons),
        update_rule=None,
    )
    network.add_connection(exc_inh_conn, source="E", target="I")
    
    # 抑制层到兴奋层的连接（全连接，固定强抑制）
    inh_exc_conn = Connection(
        source=inh_layer,
        target=exc_layer,
        w=-args.inh * (torch.ones(args.n_neurons, args.n_neurons) - torch.eye(args.n_neurons)),
        update_rule=None,
    )
    network.add_connection(inh_exc_conn, source="I", target="E")
    
    return network


def create_baseline_network(args, device):
    """
    创建自定义的空间学习网络
    
    Args:
        args: 命令行参数
        device: 计算设备
    
    Returns:
        Network: 自定义网络实例
    """
    network = Network(dt=args.dt)
    
    # 输入层（784个神经元，适配28x28图像）
    input_layer = Input(n=784, shape=(1, 28, 28), traces=True)
    # 兴奋层
    exc_layer = LIFNodes(n=args.n_neurons, traces=True, theta_plus=args.theta_plus)
    # 抑制层
    inh_layer = LIFNodes(n=args.n_neurons, traces=True)
    
    # 添加层到网络
    network.add_layer(input_layer, name="X")
    network.add_layer(exc_layer, name="E")
    network.add_layer(inh_layer, name="I")
    
    # 输入层到兴奋层的连接（可学习）
    input_exc_conn = Connection(
        source=input_layer,
        target=exc_layer,
        w=0.3 * torch.rand(784, args.n_neurons),
        update_rule=PostPre,
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
        w=args.exc * torch.eye(args.n_neurons),
        update_rule=None,
    )
    network.add_connection(exc_inh_conn, source="E", target="I")
    
    # 抑制层到兴奋层的连接（全连接，固定强抑制）
    inh_exc_conn = Connection(
        source=inh_layer,
        target=exc_layer,
        w=-args.inh * (torch.ones(args.n_neurons, args.n_neurons) - torch.eye(args.n_neurons)),
        update_rule=None,
    )
    network.add_connection(inh_exc_conn, source="I", target="E")
    
    return network


def setup_monitors(network, time, dt, device, network_type="custom"):
    """
    为网络设置监视器
    
    Args:
        network: 网络实例
        time: 仿真时间
        dt: 时间步长
        device: 计算设备
        network_type: 网络类型（"custom" 或 "baseline"）
    
    Returns:
        tuple: (spikes, voltages, voltage_monitors)
    """
    # 设置脉冲监视器
    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(
            network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
        )
        network.add_monitor(spikes[layer], name="%s_spikes" % layer)
    
    # 设置电压监视器
    voltages = {}
    voltage_monitors = {}
    
    if network_type == "custom":
        # 自定义网络使用E和I层
        exc_layer_name, inh_layer_name = "E", "I"
    else:
        # 基线网络使用Ae和Ai层
        exc_layer_name, inh_layer_name = "Ae", "Ai"
    
    for layer_name in [exc_layer_name, inh_layer_name]:
        if layer_name in network.layers:
            voltage_monitors[layer_name] = Monitor(
                network.layers[layer_name], ["v"], time=int(time / dt), device=device
            )
            network.add_monitor(voltage_monitors[layer_name], name=f"{layer_name}_voltage")
            voltages[layer_name] = voltage_monitors[layer_name]
    
    return spikes, voltages, voltage_monitors


def train_network(network, args, device, network_name="Network"):
    """
    训练网络并返回准确率
    
    Args:
        network: 要训练的网络
        args: 命令行参数
        device: 计算设备
        network_name: 网络名称（用于显示）
    
    Returns:
        dict: 包含准确率信息的字典
    """
    print(f"\n开始训练 {network_name}...")
    
    # 设置监视器
    network_type = "custom" if "自定义" in network_name else "baseline"
    spikes, voltages, voltage_monitors = setup_monitors(network, args.time, args.dt, device, network_type)
    
    # 获取兴奋层名称
    exc_layer_name = "E" if network_type == "custom" else "Ae"
    
    # 加载MNIST数据
    dataset = MNIST(
        PoissonEncoder(time=args.time, dt=args.dt),
        None,
        "../../data/MNIST",
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * args.intensity)]
        ),
    )
    
    # 神经元分配和脉冲比例
    n_classes = 10
    assignments = -torch.ones(args.n_neurons, device=device)
    proportions = torch.zeros((args.n_neurons, n_classes), device=device)
    rates = torch.zeros((args.n_neurons, n_classes), device=device)
    
    # 准确率序列
    accuracy = {"all": [], "proportion": []}
    
    update_steps = int(args.n_train / args.batch_size / args.n_updates)
    update_interval = update_steps * args.batch_size
    spike_record = torch.zeros((update_interval, int(args.time / args.dt), args.n_neurons), device=device)
    
    start = t()
    
    for epoch in range(args.n_epochs):
        labels = []
        
        if epoch % args.progress_interval == 0:
            print(f"\n{network_name} 进度: {epoch} / {args.n_epochs} ({t() - start:.4f} 秒)")
            start = t()
        
        # 创建数据加载器
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.n_workers,
            pin_memory=args.gpu,
        )
        
        pbar_training = tqdm(total=args.n_train, desc=f"{network_name} 训练")
        for step, batch in enumerate(train_dataloader):
            if step * args.batch_size > args.n_train:
                break
            
            # 分配标签给兴奋神经元
            if step % update_steps == 0 and step > 0:
                label_tensor = torch.tensor(labels, device=device)
                
                # 获取网络预测
                all_activity_pred = all_activity(
                    spikes=spike_record, assignments=assignments, n_labels=n_classes
                )
                proportion_pred = proportion_weighting(
                    spikes=spike_record,
                    assignments=assignments,
                    proportions=proportions,
                    n_labels=n_classes,
                )
                
                # 计算网络准确率
                accuracy["all"].append(
                    100
                    * torch.sum(label_tensor.long() == all_activity_pred.to(device)).item()
                    / len(label_tensor)
                )
                accuracy["proportion"].append(
                    100
                    * torch.sum(label_tensor.long() == proportion_pred.to(device)).item()
                    / len(label_tensor)
                )
                
                # 分配标签给兴奋层神经元
                assignments, proportions, rates = assign_labels(
                    spikes=spike_record,
                    labels=label_tensor,
                    n_labels=n_classes,
                    rates=rates,
                )
                
                labels = []
            
            # 获取下一个输入样本
            inputs = {"X": batch["encoded_image"]}
            if args.gpu:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 记住标签
            labels.extend(batch["label"].tolist())
            
            # 在输入上运行网络
            network.run(inputs=inputs, time=args.time)
            
            # 添加到脉冲记录
            s = spikes[exc_layer_name].get("s").permute((1, 0, 2))
            spike_record[
                (step * args.batch_size)
                % update_interval : (step * args.batch_size % update_interval)
                + s.size(0)
            ] = s
            
            network.reset_state_variables()  # 重置状态变量
            pbar_training.update(args.batch_size)
        pbar_training.close()
    
    print(f"{network_name} 进度: {epoch + 1} / {args.n_epochs} ({t() - start:.4f} 秒)")
    print(f"\n{network_name} 训练完成.\n")
    
    return accuracy, assignments, proportions, rates


def test_network(network, args, device, assignments, proportions, network_name="Network"):
    """
    测试网络并返回准确率
    
    Args:
        network: 要测试的网络
        args: 命令行参数
        device: 计算设备
        assignments: 神经元分配
        proportions: 脉冲比例
        network_name: 网络名称
    
    Returns:
        dict: 测试准确率
    """
    print(f"\n开始测试 {network_name}...")
    
    # 设置监视器
    network_type = "custom" if "自定义" in network_name else "baseline"
    spikes, _, _ = setup_monitors(network, args.time, args.dt, device, network_type)
    
    # 获取兴奋层名称
    exc_layer_name = "E" if network_type == "custom" else "Ae"
    
    # 加载MNIST测试数据
    test_dataset = MNIST(
        PoissonEncoder(time=args.time, dt=args.dt),
        None,
        root=os.path.join(ROOT_DIR, "data", "MNIST"),
        download=True,
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * args.intensity)]
        ),
    )
    
    # 创建数据加载器
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        pin_memory=args.gpu,
    )
    
    # 准确率序列
    accuracy = {"all": 0, "proportion": 0}
    n_classes = 10
    
    network.train(mode=False)
    start = t()
    
    pbar = tqdm(total=args.n_test, desc=f"{network_name} 测试")
    
    for step, batch in enumerate(test_dataloader):
        if step * args.batch_size > args.n_test:
            break
        
        # 获取下一个输入样本
        inputs = {"X": batch["encoded_image"]}
        if args.gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 在输入上运行网络
        network.run(inputs=inputs, time=args.time)
        
        # 添加到脉冲记录
        spike_record = spikes[exc_layer_name].get("s").permute((1, 0, 2))
        
        # 转换标签数组为张量
        label_tensor = torch.tensor(batch["label"], device=device)
        
        # 获取网络预测
        all_activity_pred = all_activity(
            spikes=spike_record, assignments=assignments, n_labels=n_classes
        )
        proportion_pred = proportion_weighting(
            spikes=spike_record,
            assignments=assignments,
            proportions=proportions,
            n_labels=n_classes,
        )
        
        # 计算网络准确率
        accuracy["all"] += float(
            torch.sum(label_tensor.long() == all_activity_pred.to(device)).item()
        )
        accuracy["proportion"] += float(
            torch.sum(label_tensor.long() == proportion_pred.to(device)).item()
        )
        
        network.reset_state_variables()  # 重置状态变量
        pbar.update(args.batch_size)
    pbar.close()
    
    # 计算最终准确率
    accuracy["all"] = accuracy["all"] / args.n_test * 100
    accuracy["proportion"] = accuracy["proportion"] / args.n_test * 100
    
    print(f"\n{network_name} 全活动准确率: {accuracy['all']:.2f}%")
    print(f"{network_name} 比例加权准确率: {accuracy['proportion']:.2f}%")
    
    return accuracy


def main():
    """
    主函数：运行网络对比实验
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_neurons", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--n_test", type=int, default=10000)
    parser.add_argument("--n_train", type=int, default=60000)
    parser.add_argument("--n_workers", type=int, default=-1)
    parser.add_argument("--n_updates", type=int, default=10)
    parser.add_argument("--exc", type=float, default=22.5)
    parser.add_argument("--inh", type=float, default=120)
    parser.add_argument("--theta_plus", type=float, default=0.05)
    parser.add_argument("--time", type=int, default=100)
    parser.add_argument("--dt", type=int, default=1.0)
    parser.add_argument("--intensity", type=float, default=128)
    parser.add_argument("--progress_interval", type=int, default=10)
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--test", dest="train", action="store_false")
    parser.add_argument("--plot", dest="plot", action="store_true")
    parser.add_argument("--gpu", dest="gpu", action="store_true")
    parser.set_defaults(plot=True, gpu=True)
    
    args = parser.parse_args()
    
    # 设置设备
    device = "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    else:
        torch.manual_seed(args.seed)
        device = "cpu"
        if args.gpu:
            args.gpu = False
    
    torch.set_num_threads(os.cpu_count() - 1)
    print("运行设备 = ", device)
    
    # 确定工作线程数
    if args.n_workers == -1:
        args.n_workers = 0
    
    print("\n=== 脉冲神经网络对比实验 ===")
    print(f"自定义空间学习网络 vs DiehlAndCook2015基线网络")
    print(f"神经元数量: {args.n_neurons}")
    print(f"训练样本: {args.n_train}, 测试样本: {args.n_test}")
    print(f"批次大小: {args.batch_size}, 训练轮数: {args.n_epochs}")
    
    # 创建网络
    print("\n创建网络...")
    custom_network = create_custom_network(args, device)
    baseline_network = create_baseline_network(args, device)
    
    if args.gpu:
        custom_network.to("cuda")
        baseline_network.to("cuda")
    
    # 训练网络
    if args.train:
        print("\n=== 训练阶段 ===")
        
        # 训练自定义网络
        custom_accuracy, custom_assignments, custom_proportions, custom_rates = train_network(
            custom_network, args, device, "自定义空间学习网络"
        )
        
        # 训练基线网络
        baseline_accuracy, baseline_assignments, baseline_proportions, baseline_rates = train_network(
            baseline_network, args, device, "DiehlAndCook2015基线网络"
        )
        
        # 显示训练结果对比
        print("\n=== 训练结果对比 ===")
        if custom_accuracy["all"]:
            print(f"自定义网络最终训练准确率: {custom_accuracy['all'][-1]:.2f}%")
        if baseline_accuracy["all"]:
            print(f"基线网络最终训练准确率: {baseline_accuracy['all'][-1]:.2f}%")
    
    # 测试网络
    print("\n=== 测试阶段 ===")
    
    # 如果没有训练，需要创建默认的分配
    if not args.train:
        n_classes = 10
        custom_assignments = -torch.ones(args.n_neurons, device=device)
        custom_proportions = torch.zeros((args.n_neurons, n_classes), device=device)
        baseline_assignments = -torch.ones(args.n_neurons, device=device)
        baseline_proportions = torch.zeros((args.n_neurons, n_classes), device=device)
    
    # 测试自定义网络
    custom_test_accuracy = test_network(
        custom_network, args, device, custom_assignments, custom_proportions, "自定义空间学习网络"
    )
    
    # 测试基线网络
    baseline_test_accuracy = test_network(
        baseline_network, args, device, baseline_assignments, baseline_proportions, "DiehlAndCook2015基线网络"
    )
    
    # 显示最终对比结果
    print("\n" + "="*50)
    print("最终对比结果")
    print("="*50)
    print(f"自定义空间学习网络:")
    print(f"  全活动准确率: {custom_test_accuracy['all']:.2f}%")
    print(f"  比例加权准确率: {custom_test_accuracy['proportion']:.2f}%")
    print(f"\nDiehlAndCook2015基线网络:")
    print(f"  全活动准确率: {baseline_test_accuracy['all']:.2f}%")
    print(f"  比例加权准确率: {baseline_test_accuracy['proportion']:.2f}%")
    
    # 计算改进
    all_improvement = custom_test_accuracy['all'] - baseline_test_accuracy['all']
    prop_improvement = custom_test_accuracy['proportion'] - baseline_test_accuracy['proportion']
    
    print(f"\n性能改进:")
    print(f"  全活动准确率改进: {all_improvement:+.2f}%")
    print(f"  比例加权准确率改进: {prop_improvement:+.2f}%")
    
    if all_improvement > 0:
        print(f"\n🎉 自定义空间学习网络在全活动准确率上优于基线网络!")
    elif all_improvement < 0:
        print(f"\n📉 基线网络在全活动准确率上优于自定义网络.")
    else:
        print(f"\n🤝 两个网络在全活动准确率上表现相当.")
    
    print("\n实验完成!")


if __name__ == "__main__":
    main()