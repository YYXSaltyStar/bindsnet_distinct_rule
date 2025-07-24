import argparse
import os
import pickle
import json
from datetime import datetime
from time import time as t

import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader

from bindsnet.analysis.plotting import (
    plot_conv2d_weights,
    plot_input,
    plot_spikes,
    plot_voltages,
)
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import DiehlAndCookNodes, Input
from bindsnet.network.topology import Connection, Conv2dConnection
from bindsnet.evaluation import all_activity, proportion_weighting

print()

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_train", type=int, default=60000)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--kernel_size", type=int, default=16)
parser.add_argument("--stride", type=int, default=4)
parser.add_argument("--n_filters", type=int, default=25)
parser.add_argument("--padding", type=int, default=0)
parser.add_argument("--time", type=int, default=50)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128.0)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--save_model", type=str, default=None, help="保存模型的路径")
parser.add_argument("--load_model", type=str, default="./models\conv_mnist_epoch_1_20250725_002254.pkl", help="加载模型的路径")
parser.add_argument("--save_interval", type=int, default=1, help="每隔多少个epoch保存一次模型")
parser.add_argument("--model_dir", type=str, default="./models", help="模型保存目录")
parser.set_defaults(plot=True, gpu=True, train=True)

args = parser.parse_args()

seed = args.seed
n_epochs = args.n_epochs
n_test = args.n_test
n_train = args.n_train
batch_size = args.batch_size
kernel_size = args.kernel_size
stride = args.stride
n_filters = args.n_filters
padding = args.padding
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu
save_model = args.save_model
load_model = args.load_model
save_interval = args.save_interval
model_dir = args.model_dir

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

# 创建模型保存目录
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"创建模型保存目录: {model_dir}")

def save_network_model(network, epoch, accuracy_dict=None, filename=None):
    """
    保存BindsNET网络模型
    
    Args:
        network: BindsNET网络对象
        epoch: 当前训练轮数
        accuracy_dict: 准确率字典
        filename: 保存文件名，如果为None则自动生成
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conv_mnist_epoch_{epoch}_{timestamp}.pkl"
    
    filepath = os.path.join(model_dir, filename)
    
    # 准备保存的数据
    model_data = {
        'epoch': epoch,
        'network_state': network.state_dict(),
        'network_connections': {},
        'network_layers': {},
        'training_params': {
            'batch_size': batch_size,
            'kernel_size': kernel_size,
            'stride': stride,
            'n_filters': n_filters,
            'padding': padding,
            'time': time,
            'dt': dt,
            'intensity': intensity,
            'n_train': n_train,
            'n_epochs': n_epochs
        },
        'timestamp': datetime.now().isoformat(),
        'device': str(device)
    }
    
    # 保存连接权重
    for conn_name, connection in network.connections.items():
        if hasattr(connection, 'w'):
            model_data['network_connections'][conn_name] = {
                'weights': connection.w.cpu().detach().numpy() if connection.w is not None else None,
                'connection_type': type(connection).__name__
            }
    
    # 保存层信息
    for layer_name, layer in network.layers.items():
        model_data['network_layers'][layer_name] = {
            'layer_type': type(layer).__name__,
            'shape': getattr(layer, 'shape', None),
            'n': getattr(layer, 'n', None)
        }
    
    # 保存准确率信息
    if accuracy_dict is not None:
        model_data['accuracy'] = accuracy_dict
    
    # 保存模型
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ 模型已保存到: {filepath}")
        
        # 同时保存一个JSON格式的元数据文件
        metadata_file = filepath.replace('.pkl', '_metadata.json')
        metadata = {
            'epoch': epoch,
            'training_params': model_data['training_params'],
            'timestamp': model_data['timestamp'],
            'device': model_data['device'],
            'accuracy': model_data.get('accuracy', {}),
            'model_file': filename
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"📋 元数据已保存到: {metadata_file}")
        
        return filepath
    except Exception as e:
        print(f"❌ 保存模型失败: {e}")
        return None

def load_network_model(filepath, network=None):
    """
    加载BindsNET网络模型
    
    Args:
        filepath: 模型文件路径
        network: 现有的网络对象，如果为None则需要重新构建
    
    Returns:
        tuple: (loaded_network, model_data)
    """
    try:
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"✅ 模型加载成功: {filepath}")
        print(f"📊 模型信息:")
        print(f"   - 训练轮数: {model_data['epoch']}")
        print(f"   - 保存时间: {model_data['timestamp']}")
        print(f"   - 设备: {model_data['device']}")
        
        if 'accuracy' in model_data:
            print(f"   - 准确率: {model_data['accuracy']}")
        
        # 如果提供了网络对象，加载权重
        if network is not None:
            # 加载连接权重
            for conn_name, conn_data in model_data['network_connections'].items():
                if conn_name in network.connections and conn_data['weights'] is not None:
                    weights_tensor = torch.from_numpy(conn_data['weights'])
                    if gpu and torch.cuda.is_available():
                        weights_tensor = weights_tensor.cuda()
                    # 正确设置参数：使用Parameter包装或直接设置data属性
                    if hasattr(network.connections[conn_name], 'w'):
                        if isinstance(network.connections[conn_name].w, torch.nn.Parameter):
                            network.connections[conn_name].w.data = weights_tensor
                        else:
                            network.connections[conn_name].w = torch.nn.Parameter(weights_tensor)
                    print(f"   - 已加载连接权重: {conn_name}")
        
        return network, model_data
    
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return None, None

def list_saved_models(model_dir):
    """
    列出保存的模型文件
    
    Args:
        model_dir: 模型保存目录
    """
    if not os.path.exists(model_dir):
        print(f"模型目录不存在: {model_dir}")
        return []
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    
    if not model_files:
        print(f"在 {model_dir} 中没有找到保存的模型")
        return []
    
    print(f"\n📁 在 {model_dir} 中找到 {len(model_files)} 个模型:")
    for i, model_file in enumerate(model_files, 1):
        metadata_file = os.path.join(model_dir, model_file.replace('.pkl', '_metadata.json'))
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                print(f"   {i}. {model_file}")
                print(f"      - 轮数: {metadata['epoch']}, 时间: {metadata['timestamp']}")
                if 'accuracy' in metadata and metadata['accuracy']:
                    print(f"      - 准确率: {metadata['accuracy']}")
            except:
                print(f"   {i}. {model_file} (无元数据)")
        else:
            print(f"   {i}. {model_file} (无元数据)")
    
    return model_files

if not train:
    update_interval = n_test

conv_size = int((28 - kernel_size + 2 * padding) / stride) + 1
per_class = int((n_filters * conv_size * conv_size) / 10)

# Build network.
network = Network()
input_layer = Input(n=784, shape=(1, 28, 28), traces=True)

conv_layer = DiehlAndCookNodes(
    n=n_filters * conv_size * conv_size,
    shape=(n_filters, conv_size, conv_size),
    traces=True,
)

conv_conn = Conv2dConnection(
    input_layer,
    conv_layer,
    kernel_size=kernel_size,
    stride=stride,
    update_rule=PostPre,
    norm=0.4 * kernel_size**2,
    nu=[1e-4, 1e-2],
    wmax=1.0,
)

w = torch.zeros(n_filters, conv_size, conv_size, n_filters, conv_size, conv_size)
for fltr1 in range(n_filters):
    for fltr2 in range(n_filters):
        if fltr1 != fltr2:
            for i in range(conv_size):
                for j in range(conv_size):
                    w[fltr1, i, j, fltr2, i, j] = -100.0

w = w.view(n_filters * conv_size * conv_size, n_filters * conv_size * conv_size)
recurrent_conn = Connection(conv_layer, conv_layer, w=w)

network.add_layer(input_layer, name="X")
network.add_layer(conv_layer, name="Y")
network.add_connection(conv_conn, source="X", target="Y")
network.add_connection(recurrent_conn, source="Y", target="Y")

# Voltage recording for excitatory and inhibitory layers.
voltage_monitor = Monitor(network.layers["Y"], ["v"], time=time)
network.add_monitor(voltage_monitor, name="output_voltage")

if gpu:
    network.to("cuda")

# 如果指定了加载模型，则加载预训练模型
if load_model is not None:
    if os.path.exists(load_model):
        loaded_network, loaded_data = load_network_model(load_model, network)
        if loaded_network is not None:
            network = loaded_network
            print(f"🔄 已加载预训练模型: {load_model}")
        else:
            print(f"⚠️ 加载模型失败，将使用随机初始化的网络")
    else:
        print(f"❌ 模型文件不存在: {load_model}")
        print("📋 可用的模型文件:")
        list_saved_models(model_dir)

# 列出现有的保存模型（如果有的话）
if train:
    print("\n📋 检查现有保存的模型:")
    existing_models = list_saved_models(model_dir)
    if not existing_models:
        print("   没有找到现有模型，将从头开始训练")

# Load MNIST data.
train_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    "../../data/MNIST",
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(network.layers[layer], state_vars=["v"], time=time)
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

# Train the network.
print("Begin training.\n")
start = t()

inpt_axes = None
inpt_ims = None
spike_ims = None
spike_axes = None
weights1_im = None
voltage_ims = None
voltage_axes = None

# 初始化一些必要的变量
n_workers = 0
n_classes = 10
assignments = torch.zeros(n_filters * conv_size * conv_size, dtype=torch.long)
proportions = torch.zeros(n_filters * conv_size * conv_size, n_classes)

for epoch in range(n_epochs):
    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=gpu,
    )
    pbar_training = tqdm(total=n_train)
    for step, batch in enumerate(train_dataloader):
        # Get next input sample.
        if step > n_train:
            break
        inputs = {"X": batch["encoded_image"].view(time, batch_size, 1, 28, 28)}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        label = batch["label"]

        # Run the network on the input.
        network.run(inputs=inputs, time=time)

        # Optionally plot various simulation information.
        if plot and batch_size == 1:
            image = batch["image"].view(28, 28)

            inpt = inputs["X"].view(time, 784).sum(0).view(28, 28)
            weights1 = conv_conn.w
            _spikes = {
                "X": spikes["X"].get("s").view(time, -1),
                "Y": spikes["Y"].get("s").view(time, -1),
            }
            _voltages = {"Y": voltages["Y"].get("v").view(time, -1)}

            inpt_axes, inpt_ims = plot_input(
                image, inpt, label=label, axes=inpt_axes, ims=inpt_ims
            )
            spike_ims, spike_axes = plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
            weights1_im = plot_conv2d_weights(weights1, im=weights1_im)
            voltage_ims, voltage_axes = plot_voltages(
                _voltages, ims=voltage_ims, axes=voltage_axes
            )

            plt.pause(1)

        network.reset_state_variables()  # Reset state variables.
        pbar_training.update(batch_size)
    pbar_training.close()
    
    # 在每个epoch结束后保存模型（如果设置了保存间隔）
    if train and (epoch + 1) % save_interval == 0:
        print(f"\n💾 保存第 {epoch + 1} 轮训练后的模型...")
        save_network_model(network, epoch + 1)

print("Progress: %d / %d (%.4f seconds)\n" % (n_epochs, n_epochs, t() - start))
print("Training complete.\n")

# 训练完成后保存最终模型
if train and save_model is not None:
    print("💾 保存最终训练模型...")
    final_model_path = save_network_model(network, n_epochs, filename=save_model)
    if final_model_path:
        print(f"🎉 最终模型已保存: {final_model_path}")
elif train:
    # 如果没有指定保存路径，使用默认名称保存最终模型
    print("💾 保存最终训练模型...")
    final_model_path = save_network_model(network, n_epochs)
    if final_model_path:
        print(f"🎉 最终模型已保存: {final_model_path}")

# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}


test_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root="../../data/MNIST",
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Create a dataloader to iterate and batch data
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_workers,
    pin_memory=gpu,
)

# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Test the network.
print("\nBegin testing...\n")
network.train(mode=False)
start = t()

pbar = tqdm(total=n_test)
pbar.set_description_str("Test progress: ")

for step, batch in enumerate(test_dataloader):
    if step * batch_size > n_test:
        break
    # Get next input sample.
    inputs = {"X": batch["encoded_image"].view(time, batch_size, 1, 28, 28)}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    network.run(inputs=inputs, time=time)

    # Add to spikes recording.
    spike_record = spikes["Y"].get("s").permute((1, 0, 2))

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
    accuracy["all"] += float(
        torch.sum(label_tensor.long() == all_activity_pred.to(device)).item()
    )
    accuracy["proportion"] += float(
        torch.sum(label_tensor.long() == proportion_pred.to(device)).item()
    )

    network.reset_state_variables()  # Reset state variables.
    pbar.update(batch_size)
pbar.close()

# 计算最终准确率
final_accuracy = {
    "all_activity": accuracy["all"] / n_test,
    "proportion_weighting": accuracy["proportion"] / n_test
}

print("\nAll activity accuracy: %.2f" % final_accuracy["all_activity"])
print("Proportion weighting accuracy: %.2f \n" % final_accuracy["proportion_weighting"])

print("Progress: %d / %d (%.4f seconds)" % (n_epochs, n_epochs, t() - start))
print("\nTesting complete.\n")

# 如果进行了测试，保存包含准确率的模型
if not train and save_model is not None:
    print("💾 保存测试后的模型（包含准确率信息）...")
    test_model_path = save_network_model(network, n_epochs, final_accuracy, save_model)
    if test_model_path:
        print(f"🎉 测试模型已保存: {test_model_path}")
elif not train:
    # 如果没有指定保存路径，使用默认名称保存测试模型
    print("💾 保存测试后的模型（包含准确率信息）...")
    test_model_path = save_network_model(network, n_epochs, final_accuracy)
    if test_model_path:
        print(f"🎉 测试模型已保存: {test_model_path}")
