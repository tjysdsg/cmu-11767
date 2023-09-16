import json
import os
from typing import List, Dict
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 200
mpl.rc("savefig", dpi=200)
import seaborn as sns

sns.set_theme()  # sns.set_style('whitegrid')


def latency_vs_flops(ax, data: List[dict]):
    latency = []
    flops = []
    for d in data:
        latency.append(np.mean(d['inference_time']))
        flops.append(d['FLOPs'])

    # ax.plot(latency, ops, marker='^')
    ax.scatter(latency, flops, marker='^')
    ax.set_xlabel('Inference latency (ms)')
    ax.set_ylabel('FLOPs')
    ax.set_title('Inference latency vs. FLOPs')


def flops_vs_acc(ax, data: List[dict]):
    acc = []
    flops = []
    for d in data:
        acc.append(d['acc'])
        flops.append(d['FLOPs'])

    ax.scatter(flops, acc, marker='^')
    ax.set_xlabel('FLOPs')
    ax.set_ylabel('Accuracy')
    ax.set_title('FLOPs vs. Accuracy')


def latency_vs_acc(ax, data: List[dict]):
    latency = []
    acc = []
    for d in data:
        latency.append(np.mean(d['inference_time']))
        acc.append(d['acc'])

    # ax.plot(latency, ops, marker='^')
    ax.scatter(latency, acc, marker='^')
    ax.set_xlabel('Inference latency (ms)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Inference latency vs. Accuracy')


def main():
    data = []
    for f in os.listdir('results'):
        if not f.endswith('.json'):
            continue
        with open(os.path.join('results', f)) as file:
            data.append(json.load(file))

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    axes = axes.flatten()
    flops_vs_acc(axes[0], data)
    latency_vs_acc(axes[1], data)
    latency_vs_flops(axes[2], data)

    # plt.legend()
    plt.savefig(f'viz.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
