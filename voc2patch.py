# -*- coding=utf-8 -*-
#!/usr/bin/python

import sys
import os
import shutil
import numpy as np
import xml.etree.ElementTree as ET
from lxml import etree, objectify
from PIL import Image 
import math

to_where = 'lffd'  # mmdetection or lffd
resolution = '1.5m' # 图像分辨率
# 检测框的ID起始值
START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = {"plane":1, "fighter": 2, "unknown": 3}
root_path = os.getcwd()
# original path
#checked_type = 'remove_difficult_object'
checked_type = 'Raw'
#dataset_folder = 'original_data/1.5m/linear_area_downsampled_from_0.5m/checked_data/' + checked_type
dataset_folder = 'voc_type'
train_or_test = 'train'
subset_folder = os.path.join(dataset_folder, train_or_test)
img_folder = os.path.join(root_path, subset_folder, 'images')
po = 540  # stride, 最大目标60个像素, overlap为60
ps = 600    # patch size
pixel_low_bound = 8  
# save path
save_folder = os.path.join(to_where, resolution, str(ps)+'_'+str(pixel_low_bound), checked_type, train_or_test)
ann_save_folder = os.path.join(save_folder, 'annotations')
img_save_folder = os.path.join(save_folder, 'images')
if not os.path.exists(ann_save_folder):
    os.makedirs(ann_save_folder)
if not os.path.exists(img_save_folder):
    os.mkdir(img_save_folder)

def get(root, name):
    vars = root.findall(name)

    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


# 得到图片唯一标识号
def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]
        return int(filename)
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.'%(filename))


def convert(xml_list, xml_dir):
    '''
    :param xml_list: 需要转换的XML文件列表
    :param xml_dir: XML的存储文件夹
    :return: None
    '''
    list_fp = xml_list
    
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    for line in list_fp:
        line = line.strip()
        print("buddy~ Processing {}".format(line))
        # 解析XML
        xml_f = os.path.join(xml_dir, line)
        tree = ET.parse(xml_f)
        root = tree.getroot()
        if get(root, 'filename')[0].text[-4:] != '.tif':
            ori_name = get(root, 'filename')[0].text
            img_path = os.path.join(img_folder, get(root, 'filename')[0].text)# + '.tif')
        else:
            ori_name = get(root, 'filename')[0].text[:-4]
            img_path = os.path.join(img_folder, get(root, 'filename')[0].text)
        ori_img = np.array(Image.open(img_path))
        img_size = ori_img.shape[0]
        ph = math.floor((img_size-ps)/po)+1+1
        for ii in range(ph**2):
            # ph单列patch数量，po滑动窗口步长（重叠部分），ps为patch大小
            if ii%ph==ph-1 and int(ii/ph)!=ph-1:
                patch_img = ori_img[img_size-ps:img_size, int(ii/ph)*po:int(ii/ph)*po+ps, :]
                # patch的左上角顶点坐标
                yy = img_size-ps
                xx = int(ii/ph)*po
            elif ii%ph!=ph-1 and int(ii/ph)==ph-1:
                patch_img = ori_img[ii%ph*po:ii%ph*po+ps, img_size-ps:img_size, :]
                yy = ii % ph*po
                xx = img_size-ps
            elif ii%ph==ph-1 and int(ii/ph)==ph-1:
                patch_img = ori_img[img_size-ps:img_size, img_size-ps:img_size, :]
                yy = img_size-ps
                xx = img_size-ps
            elif ii%ph!=ph-1 and int(ii/ph)!=ph-1:
                patch_img = ori_img[ii%ph*po:ii%ph*po+ps, int(ii/ph)*po:int(ii/ph)*po+ps, :]
                yy = ii % ph*po
                xx = int(ii/ph)*po  
            filename = ori_name + '_' + str(ii) + '.tif' 
            E = objectify.ElementMaker(annotate=False)
            anno_tree = E.annotation(E.folder('Annotations'),
                                    E.filename(filename),
                                    E.source(E.database('Unknow')),
                                    E.size(E.width(ps),
                                        E.height(ps),
                                        E.depth(3)),
                                    E.segmented(0))

            for obj in get(root, 'object'):
                category = get_and_check(obj, 'name', 1).text
                category_id = categories[category]
                bndbox = get_and_check(obj, 'bndbox', 1)
                xmin = int(get_and_check(bndbox, 'xmin', 1).text)
                ymin = int(get_and_check(bndbox, 'ymin', 1).text)
                xmax = int(get_and_check(bndbox, 'xmax', 1).text)
                ymax = int(get_and_check(bndbox, 'ymax', 1).text)
                # 检测框（左上角或右下角）在patch内
                if (0<xmin-xx<ps and 0<ymin-yy<ps) or \
                    (0<xmax-xx<ps and 0<ymax-yy<ps):
                    xmin_patch = xmin - xx
                    ymin_patch = ymin - yy
                    xmax_patch = xmax - xx
                    ymax_patch = ymax - yy
                    area_old = abs(xmax_patch - xmin_patch)*abs(ymax_patch - ymin_patch)
                    # 检测框超出patch时，截断
                    if ymax - yy > ps:
                        ymax_patch = ps
                    if xmax - xx > ps:
                        xmax_patch = ps
                    if ymin - yy < 0:
                        ymin_patch = 0
                    if xmin - xx < 0:
                        xmin_patch = 0
                    o_width = abs(xmax_patch - xmin_patch)
                    o_height = abs(ymax_patch - ymin_patch)
                    area = o_width*o_height
                    # 加area的条件, 是为了避免拆分batch后出现某个边只有一两个像素的情况
                    if max(o_width, o_height) < pixel_low_bound or area/area_old < 0.7:
                        continue
                    E2 = objectify.ElementMaker(annotate=False)
                    anno_tree2 = E2.object(E.name(category),
                                        E.pose('Unspecified'),
                                        E.truncated("0"),
                                        E.difficult(0),
                                        E.bndbox(E.xmin(xmin_patch),
                                                E.ymin(ymin_patch),
                                                E.xmax(xmax_patch),
                                                E.ymax(ymax_patch)))
                    anno_tree.append(anno_tree2)
                    save_ann_name = ori_name + '_' + str(ii) + '.xml'
                    ann_save = os.path.join(ann_save_folder, save_ann_name)
                    etree.ElementTree(anno_tree).write(ann_save, pretty_print=True)
                    patch_pil = Image.fromarray(patch_img) 
                    patch_pil.save(os.path.join(img_save_folder, filename))

if __name__ == '__main__':
    xml_dir = os.path.join(root_path, subset_folder, 'annotations')
    xml_labels = os.listdir(os.path.join(root_path, subset_folder, 'annotations'))
    convert(xml_labels, xml_dir)
