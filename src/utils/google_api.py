import requests


class GoogleApi:
    _API_KEY = 'AIzaSyDysZ-cHFt1yqSyMmSxyOQ23lEXBUO4vBo'

    @classmethod
    def get_places(cls, latitude: float, longitude: float, radius: float=100):
        url = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json'
        result = requests.get(url, params={
            'key': cls._API_KEY,
            'location': '{},{}'.format(latitude, longitude),
            'radius': radius,
            'rankby': 'prominence'
        })
        return result.json()['results']


if __name__ == '__main__':
    GoogleApi.get_places(47.37013738861315, 8.543545133977558)
