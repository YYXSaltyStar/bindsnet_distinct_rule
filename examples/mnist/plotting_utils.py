"""
神经网络绘图工具模块
用于绘制和保存SNN训练和测试过程中的图像
"""

import os
import matplotlib.pyplot as plt
import torch
import numpy as np


def setup_image_directories(experiment_name):
    """
    设置实验图像保存目录
    
    参数:
        experiment_name: 实验名称，用于创建相应的文件夹
        
    返回:
        directories: 包含不同类型图像保存路径的字典
    """
    # 基础目录
    base_dir = os.path.join("..", "..", "plots")
    
    # 创建实验目录
    experiment_dir = os.path.join(base_dir, experiment_name)
    
    # 各类图像目录
    directories = {
        "inputs": os.path.join(experiment_dir, "inputs"),
        "spikes": os.path.join(experiment_dir, "spikes"),
        "weights": os.path.join(experiment_dir, "weights"),
        "assignments": os.path.join(experiment_dir, "assignments"),
        "performance": os.path.join(experiment_dir, "performance"),
        "voltages": os.path.join(experiment_dir, "voltages")
    }
    
    # 创建所有目录
    for directory in directories.values():
        os.makedirs(directory, exist_ok=True)
    
    return directories


def plot_and_save_images(
    sample_idx, 
    image, 
    inpt, 
    label, 
    spikes, 
    square_weights, 
    square_assignments, 
    accuracy, 
    voltages, 
    directories, 
    update_interval, 
    time,
    inpt_axes=None, 
    inpt_ims=None, 
    spike_ims=None, 
    spike_axes=None, 
    weights_im=None, 
    assigns_im=None, 
    perf_ax=None, 
    voltage_ims=None, 
    voltage_axes=None,
    save_images=False
):
    """
    绘制并保存神经网络训练/测试过程中的图像
    
    参数:
        sample_idx: 样本索引
        image: 原始图像
        inpt: 编码后的输入
        label: 样本标签
        spikes: 各层脉冲记录
        square_weights: 格式化的权重矩阵
        square_assignments: 格式化的神经元分配
        accuracy: 准确率记录
        voltages: 各层电压记录
        directories: 图像保存目录
        update_interval: 更新间隔
        time: 时间步长
        inpt_axes, inpt_ims, ...: 绘图对象
        save_images: 是否保存图像
        
    返回:
        inpt_axes, inpt_ims, ...: 更新后的绘图对象
    """
    if inpt_axes is None:
        fig, inpt_axes = plt.subplots(1, 2, figsize=(12, 5))
        # 确保在转换为numpy前先将图像移至CPU
        img_data = image
        if hasattr(img_data, 'device') and img_data.device.type != 'cpu':
            img_data = img_data.cpu()
        img_plot = img_data.numpy()
        inpt_axes[0].set_title("Original MNIST Image")
        inpt_axes[0].imshow(img_plot, cmap="binary")
        inpt_axes[0].set_xticks([])
        inpt_axes[0].set_yticks([])
        inpt_axes[1].set_title("Encoded Image")
        # 确保在转换为numpy前先将输入移至CPU
        inpt_data = inpt
        if hasattr(inpt_data, 'device') and inpt_data.device.type != 'cpu':
            inpt_data = inpt_data.cpu()
        inpt_ims = inpt_axes[1].imshow(inpt_data, cmap="binary")
        inpt_axes[1].set_xticks([])
        inpt_axes[1].set_yticks([])
        # 处理标签可能是张量或整数的情况
        label_value = label.item() if hasattr(label, 'item') else label
        inpt_axes[1].text(
            27, 0, f"Label = {label_value}", color="white", fontsize=12
        )
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        if save_images:
            plt.savefig(os.path.join(directories["inputs"], f"input_{sample_idx}.png"))
    else:
        # 确保在转换为numpy前先将输入移至CPU
        inpt_data = inpt
        if hasattr(inpt_data, 'device') and inpt_data.device.type != 'cpu':
            inpt_data = inpt_data.cpu()
        inpt_ims.set_data(inpt_data)
        # 处理标签可能是张量或整数的情况
        label_value = label.item() if hasattr(label, 'item') else label
        inpt_axes[1].text(
            27, 0, f"Label = {label_value}", color="white", fontsize=12
        )
        
        if save_images and sample_idx % update_interval == 0:
            plt.figure(inpt_axes[0].figure.number)
            plt.savefig(os.path.join(directories["inputs"], f"input_{sample_idx}.png"))

    # 绘制脉冲图
    if spike_axes is None:
        spike_fig, spike_axes = plt.subplots(1, 2, figsize=(15, 7))
        spike_axes[0].set_title("Input Layer Spikes")
        # 确保在转换为numpy前先将张量移至CPU
        spike_data = spikes["X"].get("s").view(time, 1, 28, 28).sum(0).sum(0)
        if spike_data.device.type != 'cpu':
            spike_data = spike_data.cpu()
        spike_ims = spike_axes[0].imshow(
            spike_data.numpy(),
            cmap="binary",
        )
        spike_axes[0].set_xticks([])
        spike_axes[0].set_yticks([])
        spike_axes[1].set_title("Excitatory Layer Spikes")
        # 确保在转换为numpy前先将张量移至CPU
        exc_spike_data = spikes["Ae"].get("s").view(time, -1).sum(0).reshape(10, 10)
        if exc_spike_data.device.type != 'cpu':
            exc_spike_data = exc_spike_data.cpu()
        spike_ims = [
            spike_ims,
            spike_axes[1].imshow(
                exc_spike_data.numpy(),
                cmap="binary",
            ),
        ]
        spike_axes[1].set_xticks([])
        spike_axes[1].set_yticks([])
        
        if save_images:
            plt.savefig(os.path.join(directories["spikes"], f"spike_{sample_idx}.png"))
    else:
        # 确保在转换为numpy前先将张量移至CPU
        spike_data = spikes["X"].get("s").view(time, 1, 28, 28).sum(0).sum(0)
        if spike_data.device.type != 'cpu':
            spike_data = spike_data.cpu()
        spike_ims[0].set_data(spike_data.numpy())
        
        # 确保在转换为numpy前先将张量移至CPU
        exc_spike_data = spikes["Ae"].get("s").view(time, -1).sum(0).reshape(10, 10)
        if exc_spike_data.device.type != 'cpu':
            exc_spike_data = exc_spike_data.cpu()
        spike_ims[1].set_data(exc_spike_data.numpy())
        
        if save_images and sample_idx % update_interval == 0:
            plt.figure(spike_axes[0].figure.number)
            plt.savefig(os.path.join(directories["spikes"], f"spike_{sample_idx}.png"))

    # 绘制权重图
    if weights_im is None:
        weights_fig, weights_ax = plt.subplots(figsize=(10, 10))
        weights_ax.set_title("Input -> Excitatory Weights")
        weights_im = weights_ax.imshow(square_weights, cmap="hot_r")
        weights_ax.set_xticks([])
        weights_ax.set_yticks([])
        plt.colorbar(weights_im)
        
        if save_images:
            plt.savefig(os.path.join(directories["weights"], f"weights_{sample_idx}.png"))
    else:
        weights_im.set_data(square_weights)
        
        if save_images and sample_idx % update_interval == 0:
            plt.figure(weights_im.axes.figure.number)
            plt.savefig(os.path.join(directories["weights"], f"weights_{sample_idx}.png"))

    # 绘制神经元分配图
    if assigns_im is None:
        assigns_fig, assigns_ax = plt.subplots(figsize=(10, 10))
        assigns_ax.set_title("Neuron Class Assignments")
        assigns_im = assigns_ax.imshow(square_assignments, cmap="tab10")
        assigns_ax.set_xticks([])
        assigns_ax.set_yticks([])
        plt.colorbar(assigns_im)
        
        if save_images:
            plt.savefig(os.path.join(directories["assignments"], f"assignments_{sample_idx}.png"))
    else:
        assigns_im.set_data(square_assignments)
        
        if save_images and sample_idx % update_interval == 0:
            plt.figure(assigns_im.axes.figure.number)
            plt.savefig(os.path.join(directories["assignments"], f"assignments_{sample_idx}.png"))

    # 绘制性能图
    if perf_ax is None:
        perf_fig, perf_ax = plt.subplots(figsize=(12, 8))
        perf_ax.set_title("Network Performance")
        perf_ax.set_xlabel("Training samples")
        perf_ax.set_ylabel("Accuracy")
        perf_ax.set_ylim([0, 100])
        all_activity_plot = perf_ax.plot(
            list(range(0, len(accuracy["all"]) * update_interval, update_interval)),
            [acc for acc in accuracy["all"]],
            label="All activity",
            color="blue",
        )[0]
        proportion_plot = perf_ax.plot(
            list(range(0, len(accuracy["proportion"]) * update_interval, update_interval)),
            [acc for acc in accuracy["proportion"]],
            label="Proportion weighting",
            color="red",
        )[0]
        perf_ax.legend()
        
        if save_images:
            plt.savefig(os.path.join(directories["performance"], f"performance_{sample_idx}.png"))
    else:
        all_activity_plot = perf_ax.get_lines()[0]
        proportion_plot = perf_ax.get_lines()[1]
        
        all_activity_plot.set_xdata(
            list(range(0, len(accuracy["all"]) * update_interval, update_interval))
        )
        all_activity_plot.set_ydata([acc for acc in accuracy["all"]])
        proportion_plot.set_xdata(
            list(range(0, len(accuracy["proportion"]) * update_interval, update_interval))
        )
        proportion_plot.set_ydata([acc for acc in accuracy["proportion"]])
        
        perf_ax.set_xlim([0, len(accuracy["all"]) * update_interval])
        
        if save_images and sample_idx % update_interval == 0:
            plt.figure(perf_ax.figure.number)
            plt.savefig(os.path.join(directories["performance"], f"performance_{sample_idx}.png"))

    # 绘制电压图
    if voltage_axes is None:
        voltage_fig, voltage_axes = plt.subplots(1, 2, figsize=(15, 7))
        voltage_axes[0].set_title("Excitatory Neuron Voltages")
        # 确保在转换为numpy前先将张量移至CPU
        exc_voltage_data = voltages["Ae"].view(time, -1)
        if exc_voltage_data.device.type != 'cpu':
            exc_voltage_data = exc_voltage_data.cpu()
        voltage_ims = voltage_axes[0].imshow(
            exc_voltage_data.numpy().T, cmap="plasma"
        )
        voltage_axes[0].set_aspect("auto")
        voltage_axes[0].set_xlabel("Time")
        voltage_axes[0].set_ylabel("Neuron")
        plt.colorbar(voltage_ims, ax=voltage_axes[0])
        
        voltage_axes[1].set_title("Inhibitory Neuron Voltages")
        # 确保在转换为numpy前先将张量移至CPU
        inh_voltage_data = voltages["Ai"].view(time, -1)
        if inh_voltage_data.device.type != 'cpu':
            inh_voltage_data = inh_voltage_data.cpu()
        voltage_ims = [
            voltage_ims,
            voltage_axes[1].imshow(
                inh_voltage_data.numpy().T, cmap="plasma"
            ),
        ]
        voltage_axes[1].set_aspect("auto")
        voltage_axes[1].set_xlabel("Time")
        voltage_axes[1].set_ylabel("Neuron")
        plt.colorbar(voltage_ims[1], ax=voltage_axes[1])
        
        if save_images:
            plt.savefig(os.path.join(directories["voltages"], f"voltage_{sample_idx}.png"))
    else:
        # 确保在转换为numpy前先将张量移至CPU
        exc_voltage_data = voltages["Ae"].view(time, -1)
        if exc_voltage_data.device.type != 'cpu':
            exc_voltage_data = exc_voltage_data.cpu()
        voltage_ims[0].set_data(exc_voltage_data.numpy().T)
        
        # 确保在转换为numpy前先将张量移至CPU
        inh_voltage_data = voltages["Ai"].view(time, -1)
        if inh_voltage_data.device.type != 'cpu':
            inh_voltage_data = inh_voltage_data.cpu()
        voltage_ims[1].set_data(inh_voltage_data.numpy().T)
        
        if save_images and sample_idx % update_interval == 0:
            plt.figure(voltage_axes[0].figure.number)
            plt.savefig(os.path.join(directories["voltages"], f"voltage_{sample_idx}.png"))

    plt.pause(1e-12)
    
    return (
        inpt_axes,
        inpt_ims,
        spike_ims,
        spike_axes,
        weights_im,
        assigns_im,
        perf_ax,
        voltage_ims,
        voltage_axes,
    ) 