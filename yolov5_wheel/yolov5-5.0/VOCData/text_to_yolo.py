#!/user/bin/python3
# -*- coding: UTF-8 -*-
"""
@author:quyang
@file:text_to_yolo.py.py
@time:2022/12/26-10:13
"""
# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
from os import getcwd

sets = ['train', 'val', 'test']
classes = ["good_wheel", "bad_wheel"]  # 改为自己的类别
abs_path = os.getcwd()
print(abs_path)


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(image_id):
    in_file = open(r'C:/Users/Win10/PycharmProjects/yolov5_wheel/yolov5-5.0/VOCData/Annotations/%s.xml' % image_id, encoding='UTF-8')
    out_file = open(r'C:/Users/Win10/PycharmProjects/yolov5_wheel/yolov5-5.0/VOCData/labels/%s.txt' % image_id, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        # difficult = obj.find('Difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()
for image_set in sets:
    # 这里是绝对路径，需要根据自己的情况修改
    if not os.path.exists(r'C:/Users/Win10/PycharmProjects/yolov5_wheel/yolov5-5.0/VOCData/labels/'):
        os.makedirs(r'C:/Users/Win10/PycharmProjects/yolov5_wheel/yolov5-5.0/VOCData/labels/')
    image_ids = open(r'C:/Users/Win10/PycharmProjects/yolov5_wheel/yolov5-5.0/VOCData/ImageSets/Main/%s.txt' % (image_set)).read().strip().split()

    if not os.path.exists(r'C:/Users/Win10/PycharmProjects/yolov5_wheel/yolov5-5.0/VOCData/dataSet_path/'):
        os.makedirs(r'C:/Users/Win10/PycharmProjects/yolov5_wheel/yolov5-5.0/VOCData/dataSet_path/')

    list_file = open(r'C:/Users/Win10/PycharmProjects/yolov5_wheel/yolov5-5.0/VOCData/dataSet_path/%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write('C:/Users/Win10/PycharmProjects/yolov5_wheel/yolov5-5.0/VOCData/images/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()