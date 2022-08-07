import json
import os
from PIL import Image
import shutil
from collections import Counter

def convert(json_path, save_path, data_type='train'):
    save_ann_folder = os.path.join(save_path, "Annotations")
    if not os.path.exists(save_ann_folder):
        os.makedirs(save_ann_folder)
    save_txt_path = os.path.join(save_ann_folder, data_type + "_gt.txt")
    save_img_path = os.path.join(save_path, data_type)
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)
    fout = open(save_txt_path, 'w')
    fout_line = ''
    with open(json_path, 'r+') as f:
        allData = json.load(f)

    image_ids = []
    # 统计每张图像的box数量, 用image_id检索
    for ann in allData["annotations"]:
        if ann['category_id'] == 1:
            image_ids.append(ann['image_id'])
    boxes_num_dict = Counter(image_ids)

    pre_file_name = ''
    for ann in allData["annotations"]:
        if ann['category_id'] == 1:
            file_name = allData['images'][ann['image_id']-1]['file_name']
            if file_name != pre_file_name:
                fout_line += file_name + '\n'
                boxes_num = boxes_num_dict[ann['image_id']]
                fout_line += str(boxes_num) + '\n'
                shutil.copy(os.path.join('/Users/maliang/Desktop/dota/val_plane/final', 'images', file_name),
                         os.path.join(save_img_path, file_name))
            x1 = ann["bbox"][0]
            y1 = ann["bbox"][1]
            width = ann["bbox"][2]
            height = ann["bbox"][3]
            pre_file_name = file_name
            fout_line += ' '.join([str(x1),str(y1),str(width),str(height),'0','0','0','0','0','0']) + '\n'

    fout.write(fout_line)
    fout.close()


if __name__ == "__main__":
    train_json_path = '/Users/maliang/Desktop/dota/val_plane/final/Dota_val.json'
    #val_json_path = '/Users/maliang/Desktop/dota/val_car/final/car_test.json'
    root_path = '/Users/maliang/Desktop/dota/val_plane/final'
    save_path = os.path.join(root_path, 'lffd_type')
    convert(train_json_path, save_path, 'test')
    #convert(val_json_path, save_path, 'test')

