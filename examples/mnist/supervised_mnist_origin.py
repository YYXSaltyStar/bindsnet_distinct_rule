import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights

# 导入自定义绘图工具
from plotting_utils import setup_image_directories, plot_and_save_images
# 导入自定义SNN工具模块
from snn_utils import save_weights_evolution, save_final_weights, setup_monitors, visualize_784x100_weights
#设定了以下一系列的参数命令行解释器，default代表了默认值。之后，我们可以在命令行中定义它们的值。
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
parser.set_defaults(plot=False, gpu=True, train=True)#为三个刚才定义的自变量设置默认值。

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
    inpt_shape=(1, 28, 28),
)

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

# 初始化权重绘图变量
weights_fig = None
weights_im = None
weights_history = []  # 存储每个update_interval的权重

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
        
        # 记录权重历史
        input_exc_weights = network.connections[("X", "Ae")].w
        weights_history.append({
            'sample': i,
            'weights': input_exc_weights.clone().detach().cpu()
        })
        
        # 保存权重演变
        if plot:
            # 确保输入权重格式正确
            if input_exc_weights.shape != (784, n_neurons):
                input_exc_weights = input_exc_weights.view(784, n_neurons)
            
            # 生成可视化网格
            weight_grid = visualize_784x100_weights(input_exc_weights, n_sqrt)
            
            if weights_fig is None:
                weights_fig, ax = plt.subplots(figsize=(12, 10))
                weights_im = ax.imshow(weight_grid, cmap="hot_r", vmin=0, vmax=1)
                ax.set_title(f"权重可视化 (784×{n_neurons}) - 样本 {i}")
                plt.colorbar(weights_im, ax=ax)
                # 添加标注，说明这是100个神经元的权重图
                ax.set_xlabel(f"{n_sqrt}×{n_sqrt}神经元的输入权重矩阵")
                ax.set_ylabel(f"每个小方块是一个神经元的28×28输入权重")
            else:
                weights_im.set_data(weight_grid)
                weights_fig.suptitle(f"权重可视化 (784×{n_neurons}) - 样本 {i}")
            
            # 保存当前权重图到plots目录
            weight_img_path = os.path.join("..", "..", "plots", "weights", f"weights_{i}.png")
            os.makedirs(os.path.dirname(weight_img_path), exist_ok=True)
            weights_fig.savefig(weight_img_path)
            
            # 创建以Python文件名命名的文件夹，并保存权重演变图
            file_name = os.path.splitext(os.path.basename(__file__))[0]
            print(f"\n正在保存权重到 ..\\..\\weights_evolution\\{file_name}\\sample_{i}")
            save_weights_evolution(
                weights=input_exc_weights,
                current_samples=i,
                n_sqrt=n_sqrt,
                n_neurons=n_neurons,
                file_name=file_name,
                save_individual=(i % (5 * update_interval) == 0)  # 每5个update_interval保存单个神经元图
            )
            print(f"已保存权重演变图到 ..\\..\\weights_evolution\\{file_name}\\sample_{i}")
            
            plt.pause(1e-8)

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

    # Optionally plot various simulation information.
    if plot:
        inpt = inputs["X"].view(time, 784).sum(0).view(28, 28)
        input_exc_weights = network.connections[("X", "Ae")].w
        square_weights = get_square_weights(
            input_exc_weights.view(784, n_neurons), n_sqrt, 28
        )
        square_assignments = get_square_assignments(assignments, n_sqrt)
        voltages = {"Ae": exc_voltages, "Ai": inh_voltages}

        # 只需在第一次调用时设置目录
        if i == 0:
            directories = setup_image_directories("supervised_mnist_origin")
        
        # 使用工具函数绘制和保存图像
        inpt_axes, inpt_ims, spike_ims, spike_axes, weights_im, assigns_im, perf_ax, voltage_ims, voltage_axes = \
            plot_and_save_images(
                sample_idx=i,
                image=image.sum(1).view(28, 28),
                inpt=inpt,
                label=label,
                spikes=spikes,
                square_weights=square_weights,
                square_assignments=square_assignments,
                accuracy=accuracy,
                voltages=voltages,
                directories=directories,
                update_interval=update_interval,
                time=time,
                inpt_axes=inpt_axes,
                inpt_ims=inpt_ims,
                spike_ims=spike_ims,
                spike_axes=spike_axes,
                weights_im=weights_im,
                assigns_im=assigns_im,
                perf_ax=perf_ax,
                voltage_ims=voltage_ims,
                voltage_axes=voltage_axes,
                save_images=True
            )

    network.reset_state_variables()  # Reset state variables.
    pbar.set_description_str("Train progress: ")
    pbar.update()

print("Progress: %d / %d \n" % (n_train, n_train))
print("Training complete.\n")

# 保存最终权重到本地
if len(weights_history) > 0:
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    save_final_weights(
        weights_history=weights_history,
        current_samples=n_train,
        n_neurons=n_neurons,
        n_sqrt=n_sqrt,
        network=network,
        input_exc_weights=network.connections[("X", "Ae")].w
    )

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

# 重置绘图变量，用于测试阶段
inpt_axes = None
inpt_ims = None
spike_axes = None
spike_ims = None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes = None
voltage_ims = None

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
    
    # 可选地绘制测试过程中的信息
    if plot and step % 10 == 0:  # 每10个样本绘制一次
        # 获取输入图像和权重
        inpt = inputs["X"].view(time, 784).sum(0).view(28, 28)
        input_exc_weights = network.connections[("X", "Ae")].w
        square_weights = get_square_weights(
            input_exc_weights.view(784, n_neurons), n_sqrt, 28
        )
        square_assignments = get_square_assignments(assignments, n_sqrt)
        
        # 获取电压记录
        exc_voltages = exc_voltage_monitor.get("v")
        inh_voltages = inh_voltage_monitor.get("v")
        voltages = {"Ae": exc_voltages, "Ai": inh_voltages}
        
        # 设置测试图像目录
        if step == 0:
            test_directories = setup_image_directories(os.path.splitext(os.path.basename(__file__))[0] + "_test")
        
        # 创建模拟的准确率列表，用于绘图
        test_accuracy = {"all": [accuracy["all"] / (step+1)], "proportion": [accuracy["proportion"] / (step+1)]}
        
        # 使用工具函数绘制和保存图像
        inpt_axes, inpt_ims, spike_ims, spike_axes, weights_im, assigns_im, perf_ax, voltage_ims, voltage_axes = \
            plot_and_save_images(
                sample_idx=step,
                image=batch["encoded_image"].view(time, 784).sum(0).view(28, 28),
                inpt=inpt,
                label=batch["label"],
                spikes=spikes,
                square_weights=square_weights,
                square_assignments=square_assignments,
                accuracy=test_accuracy,
                voltages=voltages,
                directories=test_directories,
                update_interval=50,  # 测试时更频繁保存
                time=time,
                inpt_axes=inpt_axes,
                inpt_ims=inpt_ims,
                spike_ims=spike_ims,
                spike_axes=spike_axes,
                weights_im=weights_im,
                assigns_im=assigns_im,
                perf_ax=perf_ax,
                voltage_ims=voltage_ims,
                voltage_axes=voltage_axes,
                save_images=True
            )

    network.reset_state_variables()  # Reset state variables.

    pbar.set_description_str(
        f"Accuracy: {(max(accuracy['all'] ,accuracy['proportion'] ) / (step+1)):.3}"
    )
    pbar.update()

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))

print("Testing complete.\n")

# 保存测试后的权重
if not train:  # 如果是测试模式
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    print("保存测试后的权重...")
    
    # 创建保存目录
    weights_dir = os.path.join("..", "..", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    # 保存测试后的权重
    test_weights_path = os.path.join(weights_dir, f"{file_name}_weights_after_test.pt")
    test_weights = network.connections[("X", "Ae")].w
    torch.save(test_weights, test_weights_path)
    print(f"测试后的权重已保存至 {test_weights_path}")

# 绘制最终权重可视化
if plot:
    print("生成最终权重可视化...")
    
    # 获取最终权重
    final_weights = network.connections[("X", "Ae")].w
    if final_weights.shape != (784, n_neurons):
        final_weights = final_weights.view(784, n_neurons)
    
    # 创建一个更详细的最终权重可视化
    final_fig = plt.figure(figsize=(20, 16))
    
    # 添加标题
    final_fig.suptitle("最终权重可视化 (784×100)", fontsize=16)
    
    # 1. 全局权重矩阵 - 大图，使用自定义函数显示
    ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
    weight_grid = visualize_784x100_weights(final_weights, n_sqrt)
    im1 = ax1.imshow(weight_grid, cmap="hot_r", vmin=0, vmax=1)
    ax1.set_title("全局权重矩阵 (100个神经元)")
    ax1.set_xlabel(f"{n_sqrt}×{n_sqrt}神经元的输入权重矩阵")
    ax1.set_ylabel(f"每个小方块是一个神经元的28×28输入权重")
    plt.colorbar(im1, ax=ax1)
    
    # 2. 权重直方图
    ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2)
    ax2.hist(final_weights.flatten().cpu().numpy(), bins=50, color='blue', alpha=0.7)
    ax2.set_title("权重分布直方图")
    ax2.set_xlabel("权重值")
    ax2.set_ylabel("频率")
    ax2.grid(True, alpha=0.3)
    
    # 3. 选择几个有代表性的神经元权重
    neurons_to_show = [0, int(n_neurons/4), int(n_neurons/2), int(3*n_neurons/4)]
    
    for i, neuron_idx in enumerate(neurons_to_show):
        ax = plt.subplot2grid((3, 4), (2, i))
        neuron_weights = final_weights[:, neuron_idx].view(28, 28).cpu().numpy()
        im = ax.imshow(neuron_weights, cmap="hot_r", vmin=0, vmax=1)
        ax.set_title(f"神经元 #{neuron_idx} 权重 (28×28)")
        plt.colorbar(im, ax=ax)
    
    # 4. 平均权重热图
    ax3 = plt.subplot2grid((3, 4), (1, 2), colspan=2)
    avg_weights = torch.mean(final_weights, dim=1).view(28, 28).cpu()
    im3 = ax3.imshow(avg_weights, cmap="hot_r")
    ax3.set_title("平均输入权重 (28×28)")
    plt.colorbar(im3, ax=ax3)
    
    # 保存最终权重可视化
    final_path = os.path.join("..", "..", "plots", "weights", "final_weights_visualization.png")
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    final_fig.savefig(final_path, dpi=300, bbox_inches='tight')
    print(f"最终权重可视化已保存至 {final_path}")
    
    # 如果有绘图窗口，保持它们打开
    plt.show()
