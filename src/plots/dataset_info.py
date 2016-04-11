import os
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt

import constants
import plots
from clustering.meanshift import cluster_photos
from processing.path_discovery import PathFinder
from utils import file


def histogram_user_per_photo():
    with open(
            constants.LOCAL_PHOTO_CACHE.format(city='zurich'), 'r') as input_data:
        input_data.readline()
        owner_counts = defaultdict(int)
        for line in input_data:
            tuple = line.strip().split(',')
            owner = tuple[2]
            owner_counts[owner] += 1
        x = []
        for owner, count in owner_counts.items():
            if count > 100:
                continue
            x.append(count)
        ax = sns.distplot(a=x, kde=False)
        ax.set_xlabel('Photos taken')
        ax.set_ylabel('Users')
        ax.set_title('Histogram of photos per user')

        plt.savefig(os.path.join(
            constants.IMAGE_PATH, 'histogram_photos_per_user.eps'),
            bbox_inches='tight')
        plt.show()


def histogram_photos_per_cluster():
    finder = PathFinder()
    all_photos = finder.photo_storage.get_photos_for_city(city_name='zurich')
    cluster_centers, labels = cluster_photos(all_photos, '100m')
    cluster_counts = defaultdict(int)
    cluster_users = defaultdict(set)
    photo_count = 0
    for photo, label in zip(all_photos, labels):
        if label < 0:
            continue
        cluster_counts[label] += 1
        cluster_users[label].add(photo['owner'])
        photo_count += 1
    x = []
    for label, count in cluster_counts.items():
        if count > 500:
            continue
        x.append(count)
    ax = sns.distplot(a=x, kde=False)
    ax.set_xlabel('Photos taken')
    ax.set_ylabel('Clusters')
    ax.set_title('Histogram of photos per cluster')

    plt.savefig(os.path.join(
        constants.IMAGE_PATH, 'histogram_photos_per_cluster.eps'),
        bbox_inches='tight')
    plt.show()

def histogram_path_length_10():
    loaded_data = file.load_csv_data(
            constants.ALL_DATA_PATH_TPL.format(
                dataset='path_set_10'))
    lengths = []
    for subset in loaded_data:
        lengths.append(len(subset))
    x = lengths
    ax = sns.distplot(a=x, kde=False)
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Paths')
    ax.set_title(r'Histogram of the path length for $|V|=10$')

    plt.savefig(os.path.join(
        constants.IMAGE_PATH, 'histogram_path_length_10.eps'),
        bbox_inches='tight')
    plt.show()

def histogram_path_length_100():
    loaded_data = file.load_csv_data(
            constants.ALL_DATA_PATH_TPL.format(
                dataset='path_set_100'))
    lengths = []
    for subset in loaded_data:
        lengths.append(len(subset))
    x = lengths
    ax = sns.distplot(a=x, kde=False)
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Paths')
    ax.set_title(r'Histogram of the path length for $|V|=100$')

    plt.savefig(os.path.join(
        constants.IMAGE_PATH, 'histogram_path_length_100.eps'),
        bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    plots.setup()
    #histogram_photos_per_cluster()
    #histogram_user_per_photo()
    #histogram_path_length_10()
    histogram_path_length_100()
