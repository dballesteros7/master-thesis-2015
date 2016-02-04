import datetime
import os
import math
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from sklearn import mixture
from sklearn.cross_validation import KFold

import constants
from storage.photo_storage import PhotoStorage


def cluster_photos(entries: np.ndarray, n_clusters: int):
    clusterer = mixture.GMM(n_components=n_clusters)
    clusterer.fit(entries)
    return clusterer


def create_nll_plot(clusters: mixture.GMM, location_array: np.ndarray):
    x = np.linspace(0, 1)
    X, Y = np.meshgrid(x, x)

    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -clusters.score_samples(XX)[0]
    Z = Z.reshape(X.shape)

    x = np.linspace(np.min(location_array[:, 0]), np.max(location_array[:, 0]))
    y = np.linspace(np.min(location_array[:, 1]), np.max(location_array[:, 1]))

    X, Y = np.meshgrid(x, y)

    CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                     levels=np.logspace(0, 3, 10))
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    plt.scatter(location_array[:, 0], location_array[:, 1], .8)

    plt.title('Negative log-likelihood predicted by a GMM')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.axis('tight')
    plt.savefig(
        os.path.join(constants.IMAGE_PATH, 'gmm_nlog_likelihood_30.eps'),
        dpi=300, bbox_inches='tight')
    plt.savefig(
        os.path.join(constants.IMAGE_PATH, 'gmm_nlog_likelihood_30.png'),
        dpi=300, bbox_inches='tight')


def create_ellipse_plot(clusters: mixture.GMM, location_array: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    make_ellipses(clusters, ax, location_array)
    plt.scatter(location_array[:, 0], location_array[:, 1], 0.8)
    plt.title('GMM cluster ellipses')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.axis('tight')
    plt.savefig(os.path.join(constants.IMAGE_PATH, 'gmm_clusters_30.eps'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(constants.IMAGE_PATH, 'gmm_clusters_30.png'),
                dpi=300, bbox_inches='tight')


def make_ellipses(gmm, ax, location_array: np.ndarray):
    colors = cm.rainbow(np.linspace(0, 1, 30))
    for n, color in enumerate(colors):
        covar = np.copy(gmm._get_covars()[n])
        covar[0, 0] = gmm._get_covars()[n][0, 0] * math.pow(
            np.max(location_array[:, 0]) - np.min(location_array[:, 0]), 2)
        covar[1, 1] = gmm._get_covars()[n][1, 1] * math.pow(
            np.max(location_array[:, 1]) - np.min(location_array[:, 1]), 2)
        v, w = np.linalg.eigh(covar)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        means = gmm.means_[n] * (
        np.max(location_array, axis=0) - np.min(location_array,
                                                axis=0)) + np.min(
            location_array, axis=0)
        print('{}, {}'.format(means[1], means[0]))
        ell = mpl.patches.Ellipse(means, v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.9)
        ax.add_artist(ell)


def prepare_dataset(k=30):
    photo_storage = PhotoStorage()
    photos = photo_storage.get_photos_for_city('zurich')

    for photo in photos:
        photo['datetaken'] = datetime.datetime.strptime(
            photo['datetaken'], '%Y-%m-%d %H:%M:%S').date()

    location_array = np.array(
        [[float(photo['latitude']), float(photo['longitude'])] for
         photo in photos])

    scaling_factor = 1 / (np.max(location_array, axis=0) -
                      np.min(location_array, axis=0))
    offset = np.min(location_array, axis=0)

    normalized_locations = (location_array - offset) * scaling_factor

    clusters = cluster_photos(normalized_locations, k)

    logprob, responsibilities = clusters.score_samples(normalized_locations)

    with open(constants.ITEMS_FEATURE_PATH_TPL.format(
            dataset=constants.GMM_DATASET_NAME, i=1), 'w') as feature_file:
        for item in responsibilities:
            feature_file.write(','.join(str(feature) for feature in item))
            feature_file.write('\n')

    with open(constants.CLUSTER_CENTERS_DATA_PATH_TPL.format(
            dataset=constants.GMM_DATASET_NAME, k=30), 'w') as cluster_file:
        for cluster_center in clusters.means_:
            real_center = (cluster_center / scaling_factor) + offset
            cluster_file.write(','.join(str(value) for value in real_center))
            cluster_file.write('\n')

    sets = defaultdict(list)
    for index, photo in enumerate(photos):
        sets[(photo['datetaken'], photo['owner'])].append(str(index))

    with open(constants.ALL_DATA_PATH_TPL.format(
            dataset=constants.GMM_DATASET_NAME), 'w') as dataset_file:
        for key in sets:
            dataset_file.write(','.join(sets[key]))
            dataset_file.write('\n')

    data = np.array(list(sets.values()))
    kf = KFold(len(data), n_folds=10, shuffle=True)
    for idx, (train_index, test_index) in enumerate(kf):
        with open(constants.DATA_PATH_TPL.format(
                dataset=constants.GMM_DATASET_NAME,
                type='train', fold=idx + 1), 'w') as output_train, \
                open(constants.DATA_PATH_TPL.format(
                    dataset=constants.GMM_DATASET_NAME,
                    type='test', fold=idx + 1), 'w') as output_test:
            for path_set in data[train_index]:
                output_train.write(','.join(path_set) + '\n')
            for path_set in data[test_index]:
                output_test.write(','.join(path_set) + '\n')


def main():
    np.random.seed(constants.SEED)
    prepare_dataset()

    # location_array = np.array(
    #     [[float(photo['longitude']), float(photo['latitude'])] for
    #      photo in photos])
    #
    # normalized_locations = (location_array - np.min(location_array, axis=0)) / (
    # np.max(location_array, axis=0) - np.min(location_array, axis=0))
    #
    # # for k in (5, 10, 15, 20, 25, 30, 40, 50):
    # clusters = cluster_photos(normalized_locations, 30)
    # # print('{} & {}'.format(k, clusters.bic(normalized_locations)))
    # create_nll_plot(clusters, location_array)
    # create_ellipse_plot(clusters, location_array)


if __name__ == '__main__':
    main()
