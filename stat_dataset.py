# -*- coding=utf-8 -*-
#!/usr/bin/python

import sys
import os
import numpy as np
import xml.etree.ElementTree as ET
from lxml import etree, objectify
from collections import defaultdict
import json
from pandas import DataFrame
import shutil

# 检测框的ID起始值
START_BOUNDING_BOX_ID = 1
# If necessary, pre-define category and its id
PRE_DEFINE_CATEGORIES = {"plane":1, "fighter": 2, "unknown": 3}
save_and_check_tiny_object = False
root_path = os.getcwd()
#dataset_folder = 'original_data/1.5m/linear_area_downsampled_from_0.5m/checked_data/8pixel_checked/train'
#dataset_folder = 'original_data/1.5m/linear_area_downsampled_from_0.5m/train'
dataset_folder = 'original_data/Raw/test'

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
    object_distribution = defaultdict(defaultdict)
    scene_list = []
    for line in list_fp:
        line = line.strip()
        xml_f = os.path.join(xml_dir, line)
        tree = ET.parse(xml_f)
        root = tree.getroot()
        scene = get(root, 'filename')[0].text[:8]
        #print(scene)
        if scene not in scene_list:
            scene_list.append(scene)
    for scene_name in scene_list:
        object_distribution[scene_name]['<8'] = 0
        object_distribution[scene_name]['8-15'] = 0
        object_distribution[scene_name]['>15'] = 0
    object_distribution['all_scene']['<8'] = 0
    object_distribution['all_scene']['8-15'] = 0
    object_distribution['all_scene']['>15'] = 0
    object_distribution['all_scene']['all'] = 0
    object_distribution['all_scene']['max_size'] = 0
    object_distribution['all_scene']['min_size'] = 100

    scenes = []
    small_object_include = 0
    tiny_obj_file = []
    all_obj_file = []
    for line in list_fp:
        line = line.strip()
        print("buddy~ Processing {}".format(line))
        # 解析XML
        xml_f = os.path.join(xml_dir, line)
        tree = ET.parse(xml_f)
        root = tree.getroot()
        scene_name = get(root, 'filename')[0].text[:8]
        ori_name = get(root, 'filename')[0].text[:-4]
        if scene_name not in scenes:
            scenes.append(scene_name)
        small_obj_flag = False
        filename = ori_name + '.tif'
        all_obj_file.append(filename)
        size = get_and_check(root, 'size', 1)
        width = get_and_check(size, 'width', 1).text
        height = get_and_check(size, 'height', 1).text
        E = objectify.ElementMaker(annotate=False)
        anno_tree_tiny = E.annotation(E.folder('Annotations'),
                                 E.filename(filename),
                                 E.source(E.database('Unknow')),
                                 E.size(E.width(width),
                                        E.height(height),
                                        E.depth(3)),
                                 E.segmented(0))
        anno_tree_normal = E.annotation(E.folder('Annotations'),
                                 E.filename(filename),
                                 E.source(E.database('Unknow')),
                                 E.size(E.width(width),
                                        E.height(height),
                                        E.depth(3)),
                                 E.segmented(0))
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category != 'unknown':
                category_id = categories[category]
                bndbox = get_and_check(obj, 'bndbox', 1)
                xmin = int(get_and_check(bndbox, 'xmin', 1).text)
                ymin = int(get_and_check(bndbox, 'ymin', 1).text)
                xmax = int(get_and_check(bndbox, 'xmax', 1).text)
                ymax = int(get_and_check(bndbox, 'ymax', 1).text)
                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)
                if max(o_width, o_height) < 8:
                    # save tiny object
                    object_distribution[scene_name]['<8'] += 1
                    object_distribution['all_scene']['<8'] += 1
                    if save_and_check_tiny_object:
                        if scene_name not in tiny_obj_file:
                            tiny_obj_file.append(filename)
                        shutil.copy(os.path.join(root_path, dataset_folder, 'JPEGImages', filename),
                                    os.path.join(root_path, dataset_folder, 'tiny_object', filename))
                        E2 = objectify.ElementMaker(annotate=False)
                        anno_tree2 = E2.object(E.name(category),
                                            E.pose('Unspecified'),
                                            E.truncated("0"),
                                            E.difficult(0),
                                            E.bndbox(E.xmin(xmin),
                                                        E.ymin(ymin),
                                                        E.xmax(xmax),
                                                        E.ymax(ymax)))
                        anno_tree_tiny.append(anno_tree2)
                        save_ann_name = ori_name + '.xml'
                        tiny_ann_folder = os.path.join(root_path, dataset_folder, 'tiny_object')
                        ann_save = os.path.join(tiny_ann_folder, save_ann_name)
                        etree.ElementTree(anno_tree_tiny).write(ann_save, pretty_print=True)
                else:
                    # save normal object
                    if save_and_check_tiny_object:
                        E3 = objectify.ElementMaker(annotate=False)
                        anno_tree3 = E3.object(E.name(category),
                                            E.pose('Unspecified'),
                                            E.truncated("0"),
                                            E.difficult(0),
                                            E.bndbox(E.xmin(xmin),
                                                        E.ymin(ymin),
                                                        E.xmax(xmax),
                                                        E.ymax(ymax)))
                        anno_tree_normal.append(anno_tree3)
                        save_ann_name = ori_name + '.xml'
                        tiny_ann_folder = os.path.join(root_path, dataset_folder, 'normal_ann')
                        ann_save = os.path.join(tiny_ann_folder, save_ann_name)
                        etree.ElementTree(anno_tree_normal).write(ann_save, pretty_print=True)
                if 8 <= max(o_width, o_height) <= 15:
                    object_distribution[scene_name]['8-15'] += 1
                    object_distribution['all_scene']['8-15'] += 1
                    small_obj_flag = True
                if max(o_width, o_height) > 15:
                    object_distribution[scene_name]['>15'] += 1
                    object_distribution['all_scene']['>15'] += 1
                if max(o_width, o_height) >= object_distribution['all_scene']['max_size']:
                    object_distribution['all_scene']['max_size'] = max(o_width, o_height)
                if max(o_width, o_height) <= object_distribution['all_scene']['min_size']:
                    object_distribution['all_scene']['min_size'] = max(o_width, o_height)

                object_distribution['all_scene']['all'] +=  1

        if small_obj_flag:
            small_object_include += 1

    object_distribution['all_scene']['ratio_of_small_include'] = small_object_include/len(list_fp)
    object_distribution['all_scene']['image num'] = len(list_fp)
    object_distribution['all_scene']['scenes num'] = len(scenes)

    save_path = os.path.join(dataset_folder, 'stat.csv')
    df = DataFrame(object_distribution)
    df.to_csv(save_path)
    # # 导出到json
    # json_file = 'stat.json'
    # json_fp = open(json_file, 'w')
    # json_str = json.dumps(object_distribution)
    # json_fp.write(json_str)
    # json_fp.close()

def merge_tiny_and_normal():
    categories = PRE_DEFINE_CATEGORIES
    normal_dir = os.path.join(root_path, dataset_folder, 'normal_ann')
    normal_labels = os.listdir(os.path.join(root_path, dataset_folder, 'normal_ann'))
    normal_labels = [label for label in normal_labels if label[-4:] == '.xml']
    normal_file_list = [path.split('/')[-1] for path in normal_labels]

    tiny_dir = os.path.join(root_path, dataset_folder, 'tiny_object')
    tiny_labels = os.listdir(os.path.join(root_path, dataset_folder, 'tiny_object'))
    tiny_labels = [label for label in tiny_labels if label[-4:] == '.xml']
    tiny_file_list = [path.split('/')[-1] for path in tiny_labels]

    for ii, tiny_line in enumerate(tiny_labels):
        if tiny_file_list[ii] in normal_file_list:
            tiny_line = tiny_line.strip()
            tiny_xml_f = os.path.join(tiny_dir, tiny_line)
            tree = ET.parse(tiny_xml_f)
            root = tree.getroot()
            ori_name = get(root, 'filename')[0].text[:-4]
            filename = ori_name + '.tif'
            size = get_and_check(root, 'size', 1)
            width = get_and_check(size, 'width', 1).text
            height = get_and_check(size, 'height', 1).text
            E = objectify.ElementMaker(annotate=False)
            mix_anno_tree = E.annotation(E.folder('Annotations'),
                                         E.filename(filename),
                                         E.source(E.database('Unknow')),
                                         E.size(E.width(width),
                                                E.height(height),
                                                E.depth(3)),
                                         E.segmented(0))

            for obj in get(root, 'object'):
                category = get_and_check(obj, 'name', 1).text
                if category != 'unknown':
                    category_id = categories[category]
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
                    mix_anno_tree.append(anno_tree2)
                    save_ann_name = ori_name + '.xml'
                    tiny_ann_folder = os.path.join(
                        root_path, dataset_folder, 'mix_tiny')
                    ann_save = os.path.join(tiny_ann_folder, save_ann_name)
                    etree.ElementTree(mix_anno_tree).write(
                        ann_save, pretty_print=True)

            normal_line = normal_labels[ii]
            mormal_xml_f = os.path.join(normal_dir, normal_line)
            tree = ET.parse(mormal_xml_f)
            root = tree.getroot()
            for obj in get(root, 'object'):
                category = get_and_check(obj, 'name', 1).text
                if category != 'unknown':
                    category_id = categories[category]
                    bndbox = get_and_check(obj, 'bndbox', 1)
                    xmin = int(get_and_check(bndbox, 'xmin', 1).text)
                    ymin = int(get_and_check(bndbox, 'ymin', 1).text)
                    xmax = int(get_and_check(bndbox, 'xmax', 1).text)
                    ymax = int(get_and_check(bndbox, 'ymax', 1).text)
                    E3 = objectify.ElementMaker(annotate=False)
                    anno_tree3 = E3.object(E.name(category),
                                           E.pose('Unspecified'),
                                           E.truncated("0"),
                                           E.difficult(0),
                                           E.bndbox(E.xmin(xmin),
                                                    E.ymin(ymin),
                                                    E.xmax(xmax),
                                                    E.ymax(ymax)))
                    mix_anno_tree.append(anno_tree3)
                    save_ann_name = ori_name + '.xml'
                    normal_ann_folder = os.path.join(
                        root_path, dataset_folder, 'mix_tiny')
                    ann_save = os.path.join(normal_ann_folder, save_ann_name)
                    etree.ElementTree(mix_anno_tree).write(
                        ann_save, pretty_print=True)


if __name__ == '__main__':
    xml_dir = os.path.join(root_path, dataset_folder, 'Annotations')
    xml_labels = os.listdir(os.path.join(root_path, dataset_folder, 'Annotations'))
    convert(xml_labels, xml_dir)
    # merge_tiny_and_normal()
