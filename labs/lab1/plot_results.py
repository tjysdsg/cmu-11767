from typing import List, Dict
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 200
mpl.rc("savefig", dpi=200)
import seaborn as sns

sns.set_theme()
# sns.set_style('whitegrid')

infer_latency = {
    'Base': [0.30280367, 0.29592259, 0.29816307, 0.29357718, 0.30024839],
    'Deep': [0.34394025, 0.3621539, 0.33378154, 0.33759679, 0.35382546],
    'Shallow': [0.25778463, 0.26768555, 0.26778899, 0.25114633, 0.25300241],
    'Wide': [0.68692936, 0.70068452, 0.66972408, 0.69724862, 0.69621147],
    'Smaller input': [0.20641181, 0.18807466, 0.20986284, 0.21330206, 0.19609989],
}

accuracy = {
    'Base': 0.82,
    'Deep': 0.83,
    'Shallow': 0.83,
    'Wide': 0.83,
    'Smaller input': 0.82,
}

flops = {
    'Base': 7791873,
    'Deep': 8054529,
    'Shallow': 7660545,
    'Wide': 16108033,
    'Smaller input': 2823425,
}


def latency_vs_flops(ax, name: str):
    latency = []
    ops = []
    for model, values in infer_latency.items():
        latency.append(np.mean(values))
        ops.append(flops[model])

    # ax.plot(latency, ops, marker='^')
    ax.scatter(latency, ops, marker='^')
    ax.set_title(name)


if __name__ == '__main__':
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    axes = axes.flatten()
    latency_vs_flops(axes[0], 'Inference latency vs. FLOPs')

    plt.legend()
    plt.savefig(f'viz.png', bbox_inches='tight')
