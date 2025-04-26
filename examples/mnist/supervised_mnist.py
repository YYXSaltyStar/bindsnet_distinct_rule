import argparse
import os

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
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights
def main():
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
    parser.add_argument("--update_interval", type=int, default=256)
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--test", dest="train", action="store_false")
    parser.add_argument("--plot", dest="plot", action="store_true")
    parser.add_argument("--gpu", dest="gpu", action="store_true")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.set_defaults(plot=False, gpu=True, train=True)

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

    if gpu:
        network.to("cuda")

    exc_voltage_monitor = Monitor(network.layers["Ae"], ["v"], time=time, device=device)
    inh_voltage_monitor = Monitor(network.layers["Ai"], ["v"], time=time, device=device)
    network.add_monitor(exc_voltage_monitor, name="exc_voltage")
    network.add_monitor(inh_voltage_monitor, name="inh_voltage")

    dataset = MNIST(
        PoissonEncoder(time=time, dt=dt),
        None,
        root=os.path.join("..", "..", "data", "MNIST"),
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
        ),
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    spike_record = torch.zeros(update_interval, time, n_neurons, device=device)
    assignments = -torch.ones(n_neurons, device=device)
    proportions = torch.zeros(n_neurons, n_classes, device=device)
    rates = torch.zeros(n_neurons, n_classes, device=device)
    accuracy = {"all": [], "proportion": []}
    labels = torch.zeros(update_interval, dtype=torch.long, device=device)

    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
        network.add_monitor(spikes[layer], name=f"{layer}_spikes")

    print("Begin training.\n")

    pbar = tqdm(total=n_train)
    total_samples = 0
    current_record_idx = 0

    for i, datum in enumerate(dataloader):
        if total_samples >= n_train:
            break

        image = datum["encoded_image"]
        batch_labels = datum["label"]
        current_batch_size = batch_labels.shape[0]
        
        # 调整维度顺序
        image = image.permute(1, 0, 2, 3, 4)  # [time, batch_size, 1, 28, 28]
        
        # 填充标签
        start_idx = current_record_idx
        end_idx = start_idx + current_batch_size
        if end_idx > update_interval:
            overflow = end_idx - update_interval
            labels[start_idx:] = batch_labels[:current_batch_size - overflow]
            labels[:overflow] = batch_labels[current_batch_size - overflow:]
        else:
            labels[start_idx:end_idx] = batch_labels

        # 运行网络
        if gpu:
            inputs = {"X": image.cuda()}
        else:
            inputs = {"X": image}
            
        choice = np.random.choice(int(n_neurons / n_classes), size=n_clamp, replace=False)
        clamp = {"Ae": per_class * batch_labels.long() + torch.Tensor(choice).long()}
        
        network.run(inputs=inputs, time=time, clamp=clamp)

        # 记录脉冲
        batch_spikes = spikes["Ae"].get("s")  # [time, batch_size, n_neurons]
        for sample_idx in range(current_batch_size):
            record_idx = (current_record_idx + sample_idx) % update_interval
            spike_record[record_idx] = batch_spikes[:, sample_idx, :]

        current_record_idx = (current_record_idx + current_batch_size) % update_interval
        total_samples += current_batch_size
        pbar.update(current_batch_size)

        # 计算准确度
        if (total_samples % update_interval == 0) and (total_samples > 0):
            all_activity_pred = all_activity(spike_record, assignments, n_classes)
            proportion_pred = proportion_weighting(spike_record, assignments, proportions, n_classes)
            
            accuracy["all"].append(
                100 * torch.sum(labels == all_activity_pred).item() / update_interval
            )
            accuracy["proportion"].append(
                100 * torch.sum(labels == proportion_pred).item() / update_interval
            )
            
            print(f"\nAccuracy at {total_samples} samples:")
            print(f"All activity: {accuracy['all'][-1]:.2f}%")
            print(f"Proportion weighting: {accuracy['proportion'][-1]:.2f}%")
            
            # 更新标签分配
            assignments, proportions, rates = assign_labels(
                spike_record, labels, n_classes, rates
            )
            
            # 重置记录
            spike_record.zero_()
            labels.zero_()
            current_record_idx = 0

    network.reset_state_variables()
    pbar.close()
    print("Training complete.\n")

    # 测试代码
    print("Testing...\n")
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

    accuracy = {"all": 0.0, "proportion": 0.0}
    spike_record = torch.zeros(1, time, n_neurons, device=device)
    network.train(mode=False)

    pbar = tqdm(total=n_test)
    for step, batch in enumerate(test_dataset):
        if step >= n_test:
            break
            
        # 调整测试数据维度
        test_image = batch["encoded_image"].permute(1, 0, 2, 3, 4)  # [time, 1, 1, 28, 28]
        
        if gpu:
            inputs = {"X": test_image.cuda()}
        else:
            inputs = {"X": test_image}
        
        network.run(inputs=inputs, time=time)
        
        # 记录脉冲
        spike_record[0] = spikes["Ae"].get("s").squeeze()
        
        # 预测
        label = torch.tensor(batch["label"], device=device)
        all_activity_pred = all_activity(spike_record, assignments, n_classes)
        proportion_pred = proportion_weighting(spike_record, assignments, proportions, n_classes)
        
        accuracy["all"] += float(torch.sum(label == all_activity_pred).item())
        accuracy["proportion"] += float(torch.sum(label == proportion_pred).item())
        
        network.reset_state_variables()
        pbar.update(1)

    print("\nFinal Test Accuracy:")
    print(f"All activity: {100 * accuracy['all'] / n_test:.2f}%")
    print(f"Proportion weighting: {100 * accuracy['proportion'] / n_test:.2f}%")
    print("Testing complete.")

if __name__ == '__main__':
    main()