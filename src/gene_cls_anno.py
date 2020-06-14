import os
import json
from collections import defaultdict



def save_txt(file,save_dir,save_name):
    save_file = os.path.join(save_dir, save_name)
    with open(save_file, 'w') as sf:
        for val in file.values():
            val_str = ' '.join(val)
            sf.write(val_str + '\n')

def gene_cls_list(file, save_dir, save_name):
    imgs_dict={}
    cls_dict = defaultdict(list)
    for img in file['images']:
        imgs_dict[img['id']] = img['file_name']

    for anno in file['annotations']:
        if not anno['image_id'] in cls_dict:
            cls_dict[anno['image_id']].append(imgs_dict[anno['image_id']])
        if str(anno['category_id']) not in cls_dict[anno['image_id']]:
            cls_dict[anno['image_id']].append(str(anno['category_id']))
    save_txt(cls_dict,save_dir, save_name)

## annotation dir
anno_dir='/home/xingjia/Data/coco/annotations'
## Read Json files
train_json='instances_train2017.json'
val_json='instances_val2017.json'




with open(os.path.join(anno_dir, val_json), 'r') as av:
    val_file = json.load(av)
cat_info = val_file['categories']
cat_dict = defaultdict(list)
for cat_i in cat_info:
    cat_dict[cat_i['name']].extend([cat_i['name'], str(cat_i['id'])])
cat_txt = 'categories.txt'
save_txt(cat_dict, anno_dir,cat_txt)

gene_cls_list(val_file,anno_dir,'val_cls.txt')

with open(os.path.join(anno_dir,train_json),'r') as at:
    train_file = json.load(at)
gene_cls_list(train_file,anno_dir,'train_cls.txt')