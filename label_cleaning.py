# -*- coding=utf-8 -*-
#!/usr/bin/python
''' 标注过程中存在两个问题
1. xml文件名和其中filename标识不一致
2. 类别标签有误
'''

import sys
import os
import shutil
import numpy as np
import xml.etree.ElementTree as ET
from lxml import etree, objectify
from PIL import Image
import math
PRE_DEFINE_CATEGORIES = {"plane": 1, "fighter": 2, "unknown": 3}
#origin_ann_folder = 'original_data/0.5m/origin_all_samples/Annotations/'
origin_ann_folder = 'original-data/Annotations/'

def get(root, name):
    vars = root.findall(name)

    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError(
            'The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


# 得到图片唯一标识号
def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]
        return int(filename)
    except:
        raise NotImplementedError(
            'Filename %s is supposed to be an integer.' % (filename))


def convert(xml_list, xml_dir):
    list_fp = xml_list

    categories = PRE_DEFINE_CATEGORIES
    false_label_num = 0
    false_label_file = []
    for line in list_fp:
        line = line.strip()
        print("buddy~ Processing {}".format(line))
        # 解析XML
        xml_f = os.path.join(xml_dir, line)
        tree = ET.parse(xml_f)
        root = tree.getroot()
        ori_name = get(root, 'filename')[0].text[:-4]
        if line[:-4] != ori_name:
            filename = line[:-4] + '.tif'
        else:
            filename = ori_name + '.tif'
        size = get_and_check(root, 'size', 1)
        width = get_and_check(size, 'width', 1).text
        height = get_and_check(size, 'height', 1).text
        E = objectify.ElementMaker(annotate=False)
        anno_tree = E.annotation(E.folder('Annotations'),
                                    E.filename(filename),
                                    E.source(E.database('Unknow')),
                                    E.size(E.width(width),
                                        E.height(height),
                                        E.depth(3)),
                                    E.segmented(0))

        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category not in categories:
                false_label_file.append(line)
                category = 'plane'
                false_label_num += 1
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text)
            ymin = int(get_and_check(bndbox, 'ymin', 1).text)
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            E2 = objectify.ElementMaker(annotate=False)
            anno_tree2 = E2.object(E.name(category),
                                    E.pose('Unspecified'),
                                    E.truncated("0"),
                                    E.difficult(0),
                                   E.bndbox(E.xmin(xmin),
                                            E.ymin(ymin),
                                            E.xmax(xmax),
                                            E.ymax(ymax)))
            anno_tree.append(anno_tree2)
            ann_save = os.path.join(origin_ann_folder, line)
            etree.ElementTree(anno_tree).write(ann_save, pretty_print=True)
    print('false label num：' + str(false_label_num))
    print(set(false_label_file))

if __name__ == '__main__':
    root_path = os.getcwd()
    xml_dir = os.path.join(root_path, origin_ann_folder)

    xml_labels = os.listdir(os.path.join(root_path, origin_ann_folder))
    convert(xml_labels, xml_dir)
