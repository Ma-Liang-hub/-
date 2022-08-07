#%%
import sys
import os
from PIL import Image
import xml.etree.ElementTree as ET
from lxml import etree, objectify
import numpy as np
import cv2
from matplotlib import pyplot as plt
import albumentations as A

PRE_DEFINE_CATEGORIES = {"plane":1, "fighter": 2, "unknown": 3}

BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)

def visualize_bbox(img, bbox, class_id, class_idx_to_name, color=BOX_COLOR, thickness=2):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    class_name = class_idx_to_name[class_id]
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,TEXT_COLOR, lineType=cv2.LINE_AA)
    return img

def visualize(annotations, category_id_to_name):
    img = annotations['image'].copy()
    for idx, bbox in enumerate(annotations['bboxes']):
        img = visualize_bbox(img, bbox, annotations['category_id'][idx], category_id_to_name)
    plt.figure(figsize=(12, 12))
    plt.imshow(img)

def get_aug(aug, min_area=0., min_visibility=0.):
    return A.Compose(aug, bbox_params=A.BboxParams(format='coco',  min_area=min_area, min_visibility=min_visibility, label_fields=['category_id']))


def get(root, name):
    vars = root.findall(name)

    return vars

def get_and_check(root, name, length):
    vars = root.findall(name)
    #print(vars)     只包含 name 四个坐标？？？？
    #print(vars[0].text)  类别和坐标信息
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError(
            'The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars

def change_resolution(xml_list, xml_dir, img_root):
    list_fp = xml_list
    category_id_to_name = {1: "plane", 2: "fighter", 3: "unknown"}
    categories = PRE_DEFINE_CATEGORIES
    for line in list_fp:
        line = line.strip()
        print("buddy~ Processing {}".format(line))
        # 解析XML
        xml_f = os.path.join(xml_dir, line)#获取当前xml文件
        tree = ET.parse(xml_f)
        root = tree.getroot()
        ori_name = get(root, 'filename')[0].text[:-4]
        img_path = img_root + get(root, 'filename')[0].text
        filename = ori_name + '.tif'
        ori_img = np.array(cv2.imread(img_path))
        bboxes = []
        cat_ids = []
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text  
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            
            xmin = int(get_and_check(bndbox, 'xmin', 1).text)-1
            ymin = int(get_and_check(bndbox, 'ymin', 1).text)-1
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)+1
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)+1  #为什么吗要加1或减1
            assert(xmax > xmin)
            assert(ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            bboxes.append([xmin, ymin, o_width, o_height])#bbox的四个参数
            cat_ids.append(category_id)

        annotations = {'image': ori_img,
                    'bboxes': bboxes, 'category_id': cat_ids}
        # visualize(annotations, category_id_to_name)
        aug = get_aug([A.Resize(height=2000, width=2000, interpolation=cv2.INTER_AREA)])#图像变为2k×2k后进行数据增强
        augmented = aug(**annotations)#输出的是什么？？
       # print(augmented)
        # visualize(augmented, category_id_to_name)
        bboxes_int = np.trunc(augmented['bboxes']).astype(int).tolist()#转为整形列表
        aug_img = augmented['image']

        E = objectify.ElementMaker(annotate=False)#写xml文件
        anno_tree = E.annotation(E.folder('Annotations'),
                                E.filename(filename),
                                E.source(E.database('Unknow')),
                                E.size(E.width(aug_img.shape[0]),
                                        E.height(aug_img.shape[1]),
                                        E.depth(3)),
                                E.segmented(0))
        ii = 0
        for obj in get(root, 'object'):
            box_i = bboxes_int[ii]
            ii += 1
            E2 = objectify.ElementMaker(annotate=False)
            anno_tree2 = E2.object(E.name(category),
                                E.pose('Unspecified'),
                                E.truncated("0"),
                                E.difficult(0),
                                E.bndbox(E.xmin(box_i[0]),
                                            E.ymin(box_i[1]),
                                            E.xmax(box_i[0]+box_i[2]),
                                            E.ymax(box_i[1]+box_i[3])))
            anno_tree.append(anno_tree2)
            save_ann_name = ori_name + '.xml'
            ann_save = os.path.join(save_ann_folder, save_ann_name)
            etree.ElementTree(anno_tree).write(ann_save, pretty_print=True)
            
        img_save = os.path.join(save_img_folder, filename)
        cv2.imwrite(img_save, aug_img)

if __name__ == '__main__':
    root_path = os.getcwd()
    #origin_ann_folder = 'original_data/0.5m/origin_all_samples/Annotations/'
    origin_ann_folder = 'original-data/Annotations/'
    img_root = 'original-data/JPEGImages/'
    save_img_folder = 'original_data/1.5m/linear_area_downsampled_from_0.5m/JPEGImages/'
    save_ann_folder = 'original_data/1.5m/linear_area_downsampled_from_0.5m/Annotations/'#获取各种路径
    if not os.path.exists(save_img_folder):
        os.makedirs(save_img_folder)
    if not os.path.exists(save_ann_folder):
        os.makedirs(save_ann_folder)#没有则创建之
    xml_dir = os.path.join(root_path, origin_ann_folder)

    xml_list = os.listdir(os.path.join(root_path, origin_ann_folder))#获取xml文件的地址和列表
    change_resolution(xml_list, xml_dir, img_root)
