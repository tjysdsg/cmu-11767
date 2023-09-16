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


def get_or_mean(d):
    if isinstance(d, list):
        return np.mean(d)
    return d


def plot(ax, data: List[dict], label: str, x_key: str, y_key: str, annotation_key: str = None):
    x = []
    y = []
    ann = []
    for d in data:
        x.append(get_or_mean(d[x_key]))
        y.append(get_or_mean(d[y_key]))

        if annotation_key is not None:
            ann.append(d[annotation_key])

    ax.scatter(x, y, marker='^', label=label)

    if annotation_key is not None:
        for i, j, a in zip(x, y, ann):
            ax.annotate(f'{a}', (i, j), fontsize=9)

    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.legend()


def main():
    data = []
    for f in os.listdir('results'):
        if not f.endswith('.json'):
            continue

        with open(os.path.join('results', f)) as file:
            data.append(json.load(file))

    # variable widths
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes = axes.flatten()

    subset = [d for d in data if d['num_layers'] == 2 and d['vocab_size'] == 10000]
    label = 'layers=2 vocab=10000'
    plot(axes[0], subset, label, x_key='FLOPs', y_key='acc', annotation_key='hidden_size')
    plot(axes[1], subset, label, x_key='inference_time', y_key='acc', annotation_key='hidden_size')
    plot(axes[2], subset, label, x_key='inference_time', y_key='FLOPs', annotation_key='hidden_size')

    subset = [d for d in data if d['num_layers'] == 4 and d['vocab_size'] == 10000]
    label = 'layers=4 vocab=10000'
    plot(axes[0], subset, label, x_key='FLOPs', y_key='acc', annotation_key='hidden_size')
    plot(axes[1], subset, label, x_key='inference_time', y_key='acc', annotation_key='hidden_size')
    plot(axes[2], subset, label, x_key='inference_time', y_key='FLOPs', annotation_key='hidden_size')

    plt.savefig('viz_widths.png', bbox_inches='tight')
    plt.close('all')

    # variable depths
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes = axes.flatten()

    subset = [d for d in data if d['hidden_size'] == 256 and d['vocab_size'] == 10000]
    label = 'hidden=256 vocab=10000'
    plot(axes[0], subset, label, x_key='FLOPs', y_key='acc', annotation_key='num_layers')
    plot(axes[1], subset, label, x_key='inference_time', y_key='acc', annotation_key='num_layers')
    plot(axes[2], subset, label, x_key='inference_time', y_key='FLOPs', annotation_key='num_layers')

    subset = [d for d in data if d['hidden_size'] == 512 and d['vocab_size'] == 10000]
    label = 'hidden=512 vocab=10000'
    plot(axes[0], subset, label, x_key='FLOPs', y_key='acc', annotation_key='num_layers')
    plot(axes[1], subset, label, x_key='inference_time', y_key='acc', annotation_key='num_layers')
    plot(axes[2], subset, label, x_key='inference_time', y_key='FLOPs', annotation_key='num_layers')

    plt.savefig('viz_depths.png', bbox_inches='tight')
    plt.close('all')

    # variable input size
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes = axes.flatten()

    subset = [d for d in data if d['hidden_size'] == 256 and d['num_layers'] == 2]
    label = 'layers=2 hidden=256'
    plot(axes[0], subset, label, x_key='FLOPs', y_key='acc', annotation_key='vocab_size')
    plot(axes[1], subset, label, x_key='inference_time', y_key='acc', annotation_key='vocab_size')
    plot(axes[2], subset, label, x_key='inference_time', y_key='FLOPs', annotation_key='vocab_size')

    subset = [d for d in data if d['hidden_size'] == 256 and d['num_layers'] == 4]
    label = 'layers=4 hidden=256'
    plot(axes[0], subset, label, x_key='FLOPs', y_key='acc', annotation_key='vocab_size')
    plot(axes[1], subset, label, x_key='inference_time', y_key='acc', annotation_key='vocab_size')
    plot(axes[2], subset, label, x_key='inference_time', y_key='FLOPs', annotation_key='vocab_size')

    plt.savefig('viz_vocab_sizes.png', bbox_inches='tight')
    plt.close('all')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes = axes.flatten()
    label = None
    plot(axes[0], data, label, x_key='FLOPs', y_key='acc')
    plot(axes[1], data, label, x_key='inference_time', y_key='acc')
    plot(axes[2], data, label, x_key='inference_time', y_key='FLOPs')
    plt.savefig('viz_all.png', bbox_inches='tight')
    plt.close('all')


if __name__ == '__main__':
    main()
