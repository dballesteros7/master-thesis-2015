import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture

import constants
from storage.photo_storage import PhotoStorage


def cluster_photos(entries: np.ndarray, n_clusters: int):
    clusterer = mixture.GMM(n_components=n_clusters)
    clusterer.fit(entries)
    return clusterer


def main():
    photo_storage = PhotoStorage()
    photos = photo_storage.get_photos_for_city('zurich')
    np.random.seed(constants.SEED)

    location_array = np.array(
            [[float(photo['longitude']), float(photo['latitude'])] for
             photo in photos])

    normalized_locations = (location_array - np.min(location_array, axis=0)) / (np.max(location_array, axis=0) - np.min(location_array, axis=0))

    clusters = cluster_photos(normalized_locations, 10)

    denormalized_centers = clusters.means_ * (np.max(location_array, axis=0) - np.min(location_array, axis=0)) + np.min(location_array, axis=0)

    logprob, features = clusters.score_samples(normalized_locations)

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
    plt.axis('tight')
    plt.show()

if __name__ == '__main__':
    main()
