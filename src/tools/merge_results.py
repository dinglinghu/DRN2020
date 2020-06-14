import json
import os

res_dir = r'../exp/rodet'
res_dir_name = 'dota_hg_df_tv{}'

total_res = []
for i in range(8):
    res_path = os.path.join(res_dir, res_dir_name.format(i))
    with open(res_dir, 'r') as rd:
        res_json = json.load(rd)
        total_res.append(res_json)


save_dir = 'dota_hg_df_trainval'
save_json = 'results_trainval.json'
with open(os.path.join(res_dir,save_dir, save_json), 'w') as rss:
    json.dump(total_res, rss)