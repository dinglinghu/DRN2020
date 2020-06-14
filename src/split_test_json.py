import os
import json



json_dir = '../data/dota/'
test_json_name = 'voc_DOTA_testset_test.json'
splits = 4

test_json_path = os.path.join(json_dir, test_json_name)
with open(test_json_path, 'r') as tjp:
    test_json = json.load(tjp)

anno = test_json['annotations']
cat = test_json['categories']
type = test_json['type']
images = test_json['images']

test_json_model = {
    'categories':cat,
    'type':type,
    'annotations': anno
}
each_num = len(images) // splits
for i in range(splits):
    test_json_name_i = test_json_name.replace('t.','t{}s_{}.'.format(splits,i))
    start = i*each_num
    end = start + each_num if i !=(splits-1) else len(images)
    test_json_model['images'] = images[start:end]
    test_json_path_i = os.path.join(json_dir,test_json_name_i)
    with open(test_json_path_i, 'w') as tjp:
        json.dump(test_json_model,tjp)

# cnt =0
# testjson_dir = '../data/rosku_testsplitjson'
# for file in os.listdir(testjson_dir):
#     with open(os.path.join(testjson_dir,file), 'r') as fl:
#         json_i = json.load(fl)
#     cnt += len(json_i['images'])
# print(cnt)