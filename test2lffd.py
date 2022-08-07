import xml.etree.ElementTree as ET
import os
import shutil

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

def convert(xml_list, xml_dir, save_path, dataset='train'):
    list_fp = xml_list
    save_txt_path = os.path.join(save_path, "Annotations", dataset + "_gt.txt")
    fout = open(save_txt_path, 'w')
    fout_line = ''
    counter = 0
    # 标注基本结构
    for line in list_fp:
        line = line.strip()
        print("buddy~ Processing {}".format(line))
        # 解析XML
        xml_f = os.path.join(xml_dir, line)
        tree = ET.parse(xml_f)
        root = tree.getroot()
        path = get(root, 'path')

        # 取出图片名字
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, 'filename', 1).text
        else:
            raise NotImplementedError('%d paths found in %s'%(len(path), line))

        fout_line += filename + '\n'
        # 处理每个标注的检测框
        num_boxes = len(get(root, 'object'))
        fout_line += str(num_boxes) + '\n'
        for obj in get(root, 'object'):
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text)
            ymin = int(get_and_check(bndbox, 'ymin', 1).text)
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            assert(xmax > xmin)
            assert(ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            fout_line += ' '.join([str(xmin),str(ymin),str(o_width),str(o_height),'0','0','0','0','0','0']) + '\n'
        counter += num_boxes
    fout.write(fout_line)
    print(counter)
    fout.close()

if __name__ == "__main__":
    root_path = os.getcwd()
    patch_path = os.path.join(root_path, 'voc_type/test_crop')
    xml_dir = os.path.join(patch_path, 'Annotations')

    test_labels = os.listdir(os.path.join(patch_path, 'Annotations'))

    save_path = os.path.join(root_path, 'lffd_type/test_crop')
    ann_path = os.path.join(save_path, 'Annotations')
    test_path = os.path.join(save_path, 'test')
    if not os.path.exists(ann_path):
        os.makedirs(ann_path) 
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    convert(test_labels, xml_dir, save_path, 'test')
    for xml_file in test_labels:
        img_name = xml_file[:-4] + '.tif'
        shutil.copy(os.path.join(patch_path, 'JPEGImages', img_name),
                    os.path.join(test_path, img_name))    

