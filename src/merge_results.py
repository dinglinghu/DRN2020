import json
import os

res_dir = r'../exp/rodet'
res_dir_name = 'dota_drn_tv{}_model_140.pth_1'

total_res = []
for i in range(4):
    res_path = os.path.join(res_dir, res_dir_name.format(i),'results.json')
    with open(res_path, 'r') as rd:
        res_json = json.load(rd)
        for det in res_json:
            del det['scale']
        total_res.extend(res_json)


save_dir = 'dota_hg_drn_s1_retrain_05'
save_json = 'results_drn_s1_140.json'
with open(os.path.join(res_dir,save_dir, save_json), 'w') as rss:
    json.dump(total_res, rss)
