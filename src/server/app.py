from bson import ObjectId
from flask import Flask
from flask import jsonify
from flask import request
from flask.json import JSONEncoder
from flask.ext.cors import CORS

from clustering.meanshift import cluster_photos
from processing.path_discovery import PathFinder
from storage.cluster_storage import ClusterStorage
from storage.data_loader import DataLoader


class BSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return JSONEncoder.default(self, obj)

app = Flask(__name__)
app.json_encoder = BSONEncoder
CORS(app, expose_headers='X-JSON-Prefix')


@app.route('/cluster')
def cluster_handler():
    args = request.args
    bandwidth = args['bandwidth']
    city_name = args['city_name']
    min_cluster_photos = int(args['min_cluster_photos'])
    min_unique_users = int(args['min_unique_users'])

    storage = ClusterStorage()
    result = storage.get_cluster(city_name, bandwidth)
    if result is None:
        loader = DataLoader()
        entries = loader.load_entries(city_name)
        cluster_result = cluster_photos(entries, bandwidth)
        entries.rewind()
        result = storage.insert_cluster(city_name, bandwidth, entries, *cluster_result)

    result['clusters'] = [cluster for cluster in result['clusters'] if
                          cluster['number_of_photos'] > min_cluster_photos and
                          cluster['unique_users'] > min_unique_users]
    return jsonify(**result)


@app.route('/path')
def path_handler():
    args = request.args
    bandwidth = args['bandwidth']
    city_name = args['city_name']
    path_finder = PathFinder()
    paths = path_finder.find_random_path(city_name, bandwidth)
    colors = ['blue', 'red', 'black', 'yellow', 'green', 'brown', 'white', 'magenta', 'cyan', 'purple']
    augmented_paths = []
    for path, color in zip(paths, colors):
        augmented_paths.append({
            'points': path,
            'color': color
        })
    return jsonify(paths=augmented_paths)

if __name__ == "__main__":
    app.run(port=9999, debug=True)
