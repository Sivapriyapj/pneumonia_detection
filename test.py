import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
data = {'url' : 'https://raw.githubusercontent.com/Sivapriyapj/pneumonia_detection/master/data/data_xray/test/PNEUMONIA/person100_bacteria_475.jpeg'}
result = requests.post(url, json=data).json()
print(result)