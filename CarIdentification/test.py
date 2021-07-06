import requests
url = 'http://miriteam.ddns.net/'
files = {'file': ('1.jpg', open('uploads/1.jpg', 'rb'), 'image/jpeg')}
r = requests.post(url, files=files)
print(r.json())