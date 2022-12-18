import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
data = {'url' : 'https://github.com/Sivapriyapj/pneumonia_detection/blob/main/data/data_xray/test/PNEUMONIA/person100_bacteria_475.jpeg'}
result = requests.post(url, json=data).json()
print(result)
