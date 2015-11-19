from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from storage.cluster_storage import ClusterStorage
from storage.path_storage import PathStorage
from storage.photo_storage import PhotoStorage


def photo_per_users_graph():
    photo_storage = PhotoStorage()
    results = photo_storage.collection.aggregate([{
        '$group': {
            '_id': '$owner',
            'photos': {
                '$sum': 1
            }
        }
    }])
    results = list(results)
    samples = np.array([result['photos'] for result in results if result['photos'] < 100])
    plt.hist(samples, bins=100, color='green')
    plt.xlabel('# of photos')
    plt.ylabel('# of users')
    plt.title(r'Histogram of photos per users ($photos < 100$)')
    plt.subplots_adjust(left=0.15)
    plt.savefig('/local/workspace/master-thesis-2015/images/photos_per_user_100.png', dpi=300,
                papertype='letter', format='png')
    plt.close()
    samples = np.array([result['photos'] for result in results if 100 <= result['photos'] < 1000])
    plt.hist(samples, bins=50, color='green')
    plt.xlabel('# of photos')
    plt.ylabel('# of users')
    plt.title(r'Histogram of photos per users ($100 \leq photos < 1000$)')
    plt.subplots_adjust(left=0.15)
    plt.savefig('/local/workspace/master-thesis-2015/images/photos_per_user_100_1000.png', dpi=300,
                papertype='letter', format='png')
    plt.close()
    samples = np.array([result['photos'] for result in results if 1000 <= result['photos']])
    plt.hist(samples, bins=10, color='green')
    plt.xlabel('# of photos')
    plt.ylabel('# of users')
    plt.title(r'Histogram of photos per users ($photos \geq 1000$)')
    plt.subplots_adjust(left=0.15)
    plt.savefig('/local/workspace/master-thesis-2015/images/photos_per_user_1000.png', dpi=300,
                papertype='letter', format='png')


def photos_per_cluster_graph():
    cluster_storage = ClusterStorage()
    results = cluster_storage.collection.find({
        'bandwidth': '100m',
        'city_name': 'zurich'
    }, projection=['number_of_photos', 'unique_users'])
    results = list(results)
    samples = np.array([result['number_of_photos'] for result in results])
    plt.hist(samples, bins=100, color='blue')
    plt.xlabel('# of photos')
    plt.ylabel('# of clusters')
    plt.title(r'Histogram of photos per cluster ($b = 100m$)')
    plt.subplots_adjust(left=0.15)
    plt.savefig('/local/workspace/master-thesis-2015/images/photos_per_cluster_100m.png', dpi=300,
                papertype='letter', format='png')
    plt.close()
    samples = np.array([result['unique_users'] for result in results])
    plt.hist(samples, bins=100, color='blue')
    plt.xlabel('# of users')
    plt.ylabel('# of clusters')
    plt.title(r'Histogram of users per cluster ($b = 100m$)')
    plt.subplots_adjust(left=0.15)
    plt.savefig('/local/workspace/master-thesis-2015/images/users_per_cluster_100m.png', dpi=300,
                papertype='letter', format='png')
    plt.close()


def clusters_per_path_graph():
    path_storage = PathStorage()
    results = path_storage.collection.find({
        'bandwidth': '100m',
        'city_name': 'zurich',
        'min_unique_users': 50,
        'min_cluster_photos': 100
    })
    results = list(results)
    samples = np.array([result['number_of_clusters'] for result in results])
    plt.hist(samples, bins=40, color='red')
    plt.xlabel('# of clusters')
    plt.ylabel('# of paths')
    plt.title(r'Cluster distribution ($b = 100m$, $min\_photos = 100$, $min\_users = 50$)')
    plt.subplots_adjust(left=0.15)
    plt.savefig('/local/workspace/master-thesis-2015/images/clusters_per_path_100m_50_100.png', dpi=300,
                papertype='letter', format='png')
    plt.close()


def paths_per_cluster_graph():
    path_storage = PathStorage()
    results = path_storage.collection.find({
        'bandwidth': '100m',
        'city_name': 'zurich',
        'min_unique_users': 50,
        'min_cluster_photos': 100
    })
    clusters = defaultdict(int)
    for result in results:
        for cluster in result['clusters']:
            clusters[cluster] += 1
    samples = np.array(list(clusters.values()))
    plt.hist(samples, bins=250, color='green')
    plt.xlabel('# of paths')
    plt.ylabel('# of clusters')
    plt.title(r'Cluster coverage ($b = 100m$, $min\_photos = 100$, $min\_users = 50$)')
    plt.subplots_adjust(left=0.15)
    plt.savefig('/local/workspace/master-thesis-2015/images/paths_per_cluster_100m_50_100.png', dpi=300,
                papertype='letter', format='png')
    plt.close()

if __name__ == '__main__':
    clusters_per_path_graph()
    paths_per_cluster_graph()
