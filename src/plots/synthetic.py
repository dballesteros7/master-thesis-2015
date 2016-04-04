import os

import numpy as np

import constants
import plots
from models.general_features import GeneralFeatures

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


def plot_weights_synthetic_2():
    model = GeneralFeatures(n_items=4, features=np.identity(4),
                            l_dims=2, k_dims=2)

    model.a_weights = np.array([0, 0, 0, 0])
    model.b_weights = np.array([[10, 0], [0, 10], [10, 0], [0, 10]])
    model.c_weights = np.array([[10, 0], [10, 0], [0, 10], [0, 10]])

    model.update_composite_parameters()

    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 2, height_ratios=[1], width_ratios=[1, 1])

    cmap = sns.cubehelix_palette(8, start=1.8, light=.8, as_cmap=True)

    ax1 = fig.add_subplot(gs[0])
    color_set = ax1.matshow(model.diversity_weights, interpolation='none',
                            cmap=cmap, vmin=0, vmax=10)
    ax1.grid(None)
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(['1', '2', '3', '4'])
    ax1.set_ylabel('Item ($i$)')
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels([])
    ax1.set_xlabel('$d$')
    ax1.set_title(r'$\mathbf{W}^{b}$')
    ax1.plot([-.5,1.5], [1.5, 1.5], color='white', linestyle='--', linewidth=2)

    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    color_set = ax2.matshow(model.coherence_weights, interpolation='none',
                            cmap=cmap, vmin=0, vmax=10)
    ax2.grid(None)
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([])
    ax2.set_xlabel('$c$')
    ax2.set_title(r'$\mathbf{W}^{e}$')
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.plot([-.5,1.5], [1.5, 1.5], color='white', linestyle='--', linewidth=2)

    ax1.set_adjustable('box-forced')
    ax2.set_adjustable('box-forced')

    plt.colorbar(color_set)
    plt.savefig(os.path.join(
        constants.IMAGE_PATH, 'fldc_toy_example_mixed_weights.eps'),
        bbox_inches='tight')
    plt.show()


def plot_weights_synthetic_2_learned():
    model = GeneralFeatures(n_items=4, features=np.identity(4),
                            l_dims=2, k_dims=2)
    model.load_from_file(constants.NCE_OUT_GENERAL_PATH_TPL.format(
            dataset='path_set_synthetic_2', fold=1, l_dim=2, k_dim=2,
            index='0'))

    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 3, height_ratios=[1], width_ratios=[1, 2, 2])

    cmap = sns.cubehelix_palette(8, start=1.8, light=.8, as_cmap=True)

    ax1 = fig.add_subplot(gs[0])
    color_set = ax1.matshow(model.utilities.reshape(4, 1), interpolation='none',
                            cmap=cmap, vmin=-3, vmax=6)
    ax1.grid(None)
    ax1.set_xticks([0])
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_xticklabels([])
    ax1.set_yticklabels(['1', '2', '3', '4'])
    ax1.set_ylabel('Item ($i$)')
    ax1.set_title(r'$\mathbf{u}$')

    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    color_set = ax2.matshow(model.diversity_weights, interpolation='none',
                            cmap=cmap, vmin=-3, vmax=6)
    ax2.grid(None)
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([])
    ax2.set_xlabel('$d$')
    ax2.set_title(r'$\mathbf{W}^{b}$')
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.plot([-.5,1.5], [1.5, 1.5], color='white', linestyle='--', linewidth=2)

    ax3 = fig.add_subplot(gs[2], sharey=ax1)
    color_set = ax3.matshow(model.coherence_weights, interpolation='none',
                            cmap=cmap, vmin=-3, vmax=6)
    ax3.grid(None)
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels([])
    ax3.set_xlabel('$c$')
    ax3.set_title(r'$\mathbf{W}^{e}$')
    plt.setp(ax3.get_yticklabels(), visible=False)
    ax3.plot([-.5,1.5], [1.5, 1.5], color='white', linestyle='--', linewidth=2)

    ax1.set_adjustable('box-forced')
    ax2.set_adjustable('box-forced')
    ax3.set_adjustable('box-forced')

    plt.colorbar(color_set)
    plt.savefig(os.path.join(
        constants.IMAGE_PATH, 'fldc_toy_example_learned_weights.eps'),
        bbox_inches='tight')
    plt.show()


def plot_weights_synthetic_3():
    features = np.array([
        [4., 1., 0.],
        [4., 1., 1.],
        [3., 0., 1.],
        [3., 1., 0.],
        [2., 1., 1.],
        [2., 1., 0.],
        # [5., 0., 1.],
    ])
    model = GeneralFeatures(n_items=6, features=features,
                            l_dims=2, k_dims=1)

    model.a_weights = np.array([0.1, 0, 0])
    model.b_weights = np.array([[0, 0], [10, 0], [0, 10]])
    model.c_weights = np.array([[0.5], [0], [0]])

    model.update_composite_parameters()

    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 3, height_ratios=[1], width_ratios=[1, 1, 2])

    cmap = sns.cubehelix_palette(8, start=1.8, light=.8, as_cmap=True)

    ax1 = fig.add_subplot(gs[0])
    color_set = ax1.matshow(model.a_weights.reshape(3, 1), interpolation='none',
                            cmap=cmap, vmin=0, vmax=1)
    ax1.grid(None)
    ax1.set_xticks([0])
    ax1.set_yticks([0, 1, 2])
    ax1.set_xticklabels([])
    ax1.set_yticklabels(['1', '2', '3'])
    ax1.set_xlabel('')
    ax1.set_ylabel('Feature ($m$)')
    ax1.set_title(r'$\mathbf{a}$')

    ax2 = fig.add_subplot(gs[2], sharey=ax1)
    color_set = ax2.matshow(model.b_weights, interpolation='none',
                            cmap=cmap, vmin=0, vmax=10)
    ax2.grid(None)
    ax2.set_yticks([0, 1, 2])
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([])
    ax2.set_xlabel('$d$')
    ax2.set_title(r'$\mathbf{B}$')
    plt.setp(ax2.get_yticklabels(), visible=False)

    plt.colorbar(color_set, ax=ax2)

    ax3 = fig.add_subplot(gs[1], sharey=ax1)
    color_set = ax3.matshow(model.c_weights, interpolation='none',
                            cmap=cmap, vmin=0, vmax=1)
    ax3.grid(None)
    ax3.set_yticks([0, 1, 2])
    ax3.set_xticks([0])
    ax3.set_xticklabels([])
    ax3.set_xlabel('$c$')
    ax3.set_title(r'$\mathbf{E}$')
    plt.setp(ax3.get_yticklabels(), visible=False)

    plt.colorbar(color_set, ax=ax3)

    ax1.set_adjustable('box-forced')
    ax2.set_adjustable('box-forced')
    ax3.set_adjustable('box-forced')


    plt.savefig(os.path.join(
        constants.IMAGE_PATH, 'ffldc_toy_example.eps'),
        bbox_inches='tight')
    plt.show()


def plot_weights_synthetic_3_learned():
    features = np.array([
        [4., 1., 0.],
        [4., 1., 1.],
        [3., 0., 1.],
        [3., 1., 0.],
        [2., 1., 1.],
        [2., 1., 0.],
        # [5., 0., 1.],
    ])
    model = GeneralFeatures(n_items=6, features=features,
                            l_dims=2, k_dims=1)
    model.load_from_file(constants.NCE_OUT_GENERAL_PATH_TPL.format(
            dataset='path_set_synthetic_4', fold=1, l_dim=2, k_dim=1,
            index='1'))
    model.update_composite_parameters()

    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 3, height_ratios=[1], width_ratios=[1, 1, 2])

    cmap = sns.cubehelix_palette(8, start=1.8, light=0.8, as_cmap=True)

    ax1 = fig.add_subplot(gs[0])
    color_set = ax1.matshow(model.a_weights.reshape(3, 1), interpolation='none',
                            cmap=cmap, vmin=-1, vmax=1)
    ax1.grid(None)
    ax1.set_xticks([0])
    ax1.set_yticks([0, 1, 2])
    ax1.set_xticklabels([])
    ax1.set_yticklabels(['1', '2', '3'])
    ax1.set_xlabel('')
    ax1.set_ylabel('Feature ($m$)')
    ax1.set_title(r'$\mathbf{a}$')

    ax2 = fig.add_subplot(gs[2], sharey=ax1)
    color_set = ax2.matshow(model.b_weights, interpolation='none',
                            cmap=cmap, vmin=0, vmax=8)
    ax2.grid(None)
    ax2.set_yticks([0, 1, 2])
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([])
    ax2.set_xlabel('$d$')
    ax2.set_title(r'$\mathbf{B}$')
    plt.setp(ax2.get_yticklabels(), visible=False)

    plt.colorbar(color_set, ax=ax2)

    ax3 = fig.add_subplot(gs[1], sharey=ax1)
    color_set = ax3.matshow(model.c_weights, interpolation='none',
                            cmap=cmap, vmin=-1, vmax=1)
    ax3.grid(None)
    ax3.set_yticks([0, 1, 2])
    ax3.set_xticks([0])
    ax3.set_xticklabels([])
    ax3.set_xlabel('$c$')
    ax3.set_title(r'$\mathbf{E}$')
    plt.setp(ax3.get_yticklabels(), visible=False)

    plt.colorbar(color_set, ax=ax3)

    ax1.set_adjustable('box-forced')
    ax2.set_adjustable('box-forced')
    ax3.set_adjustable('box-forced')


    plt.savefig(os.path.join(
        constants.IMAGE_PATH, 'ffldc_toy_example_learned_weights.eps'),
        bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    plots.setup()
    plot_weights_synthetic_2()
