import requests

house = {
    'area_living': 100,
    'area_land': 100,
    'n_rooms': 1,
    'year': 2000,
    'price': 200000,
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=house, timeout=10)
print(response.json())
