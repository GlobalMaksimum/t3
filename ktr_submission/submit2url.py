# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3.7 MMDetect
#     language: python
#     name: mmdetect
# ---

# +
import json

from tqdm import tqdm

# +
f_yaya = open('yaya_json.json')
yaya_json = json.load(f_yaya)

f_arac = open('arac_json.json')
arac_json = json.load(f_arac)
# -

arac_json = {a['frame_id']: a for a in arac_json}
yaya_json = {a['frame_id']: a for a in yaya_json}

merged_json = []
for k in arac_json.keys():
    a = {
        'frame_id': k,
        'objeler': arac_json[k]['objeler'] + yaya_json[k]['objeler']
    }
    merged_json.append(a)

# +
import json
f = open('merged.json', 'w')
json.dump(merged_json, f)

f.close()
# -



import json
import requests
s = requests.Session()
url = 'http://212.68.57.202:52196/api/giris'
data = {"kadi" : "globalmaksimum","sifre" : "teknoyz442"}
r = s.post(url, json=data)
print(r.status_code)

rrr= []
for i in tqdm(merged_json):
    r = s.post("http://212.68.57.202:52196/api/cevap_gonder", json=i)
    rrr.append(r.status_code)
    if r.status_code == 200:
        print(i['frame_id'])
        print(r.status_code)


