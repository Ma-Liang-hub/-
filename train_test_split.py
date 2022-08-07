#%%
import os
from collections import Counter
import shutil

root_path = os.getcwd()
#dataset_folder = 'original_data/1.5m/linear_area_downsampled_from_0.5m/checked_data/remove_difficult_object'
dataset_folder = 'original_data/1.5m/linear_area_downsampled_from_0.5m/'
#dataset_folder = 'original_data/Raw/'

img_folder = os.path.join(root_path, dataset_folder, 'JPEGImages')
xml_dir = os.path.join(root_path, dataset_folder, 'Annotations')
xml_labels = os.listdir(os.path.join(root_path, dataset_folder, 'Annotations'))
scene_list = [path.split('/')[-1][:5] for path in xml_labels]
scene_count = Counter(scene_list)
print(scene_count)

test_scene_list = ['01_04', '01_05', '01_06', '01_12', '01_16', '01_18', '01_20', '01_22',
                  '01_26', '01_27', '01_30', '01_34', '01_35', '01_37', '01_39', '01_42',
                  '01_44', '01_47', '01_49', '02_01', '04_01', '05_04']

test_half_flag = [True, True, False, True, False, True, False, True,
                  True, True, False, True, False, False, False, False, 
                  False, False, False, False, False, False]

#%%
test_num = {}
for test_scene in test_scene_list:
    test_num[test_scene] = 0

test_labels = []
train_labels = []
for label_file in xml_labels:
    scene = label_file[:5]  #获取名字的前五位，代表场景信息
    if scene in test_scene_list:
        idx = test_scene_list.index(label_file[:5])
        if test_half_flag[idx]:
            if test_num[scene] < scene_count[scene]/2:
                test_num[scene] += 1
                test_labels.append(label_file)
            else:
                train_labels.append(label_file)
        else:
            train_labels.append(label_file)#?????原先是test_label....
    else:
        train_labels.append(label_file)

#%%
save_path = os.path.join(root_path, dataset_folder)
train_path = os.path.join(save_path, 'train')
test_path = os.path.join(save_path, 'test')

for xml_file in train_labels:
    img_name = xml_file[:-4] + '.tif'
    train_img_folder = os.path.join(train_path, 'JPEGImages')
    train_ann_folder = os.path.join(train_path, 'Annotations')
    if not os.path.exists(train_img_folder):
        os.makedirs(train_img_folder)
    if not os.path.exists(train_ann_folder):
        os.makedirs(train_ann_folder)
    shutil.copy(os.path.join(img_folder, img_name),
                os.path.join(train_img_folder, img_name))
    shutil.copy(os.path.join(xml_dir, xml_file),
                os.path.join(train_ann_folder, xml_file))

for xml_file in test_labels:
    img_name = xml_file[:-4] + '.tif'
    test_img_folder = os.path.join(test_path, 'JPEGImages')
    test_ann_folder = os.path.join(test_path, 'Annotations')
    if not os.path.exists(test_img_folder):
        os.makedirs(test_img_folder)
    if not os.path.exists(test_ann_folder):
        os.makedirs(test_ann_folder)
    shutil.copy(os.path.join(img_folder, img_name),
                os.path.join(test_img_folder, img_name))
    shutil.copy(os.path.join(xml_dir, xml_file),
                os.path.join(test_ann_folder, xml_file))


# %%
