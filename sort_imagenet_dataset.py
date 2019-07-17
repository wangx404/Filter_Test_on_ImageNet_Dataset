#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:00:48 2019

@author: wangxin
"""

import os, shutil


def sort_imagenet_1(input_dir, output_dir, label_txt="ILSVRC2013_clsloc_validation_ground_truth.txt"):
    """
    将图片文件移动至其归属的文件夹中。这里所使用的标签文本为*_ground_truth.txt。
    Move image file to sub folder it belongs to. Label file used here 
    is ILSVRC2013_clsloc_validation_ground_truth.txt.

    :param input_dir: imagenet dataset dir, str
    :param output_dir: where to store reorganized dataset, str
    :param label_txt: label file, str
    :return None:
    """
    image_list = os.listdir(input_dir)
    image_list.sort()
    
    with open(label_txt, "r") as f:
        labels = f.readlines()
    labels = [label.rstrip() for label in labels]
    
    for image, label in zip(image_list, labels):
        if not os.path.exists(os.path.join(output_dir, label)):
            os.makedirs(os.path.join(output_dir, label))
        shutil.move(os.path.join(input_dir, image), 
                    os.path.join(output_dir, label, image))
    
    print("finished.")


def sort_imagenet_2(input_dir, output_dir, label_txt):
    """
    将图片文件移动至其归属的文件夹中。这里所使用的标签文本为val.txt。
    Move image file to sub folder it belongs to. Label file used here 
    is val.txt.

    :param input_dir: imagenet dataset dir, str
    :param output_dir: where to store reorganized dataset, str
    :param label_txt: label file, str
    :return None:
    """
    with open(label_txt, "r") as f:
        labels = f.readlines()
    labels = [label.rstrip() for label in labels]
    
    for label in labels:
        image, sub_dir = label.split(" ")
        if not os.path.exists(os.path.join(output_dir, sub_dir)):
            os.makedirs(os.path.join(output_dir, sub_dir))
    
        shutil.move(os.path.join(input_dir, image), 
                    os.path.join(output_dir, sub_dir, image))

    print("Finished.")


def rename_folders(input_dir):
    """
    将imagenet中的文件夹重命名为"03d"的形式。
    Rename folders of imagenet into the format of "03d".

    :param input_dir: reorganized imagenet folder, str
    :return None:
    """
    for index in range(0, 1000):
        sub_dir = str(index)
        new_sub_dir = "%03d" % index
        os.rename(os.path.join(input_dir, sub_dir), 
                os.path.join(input_dir, new_sub_dir))


if __name__ == "__main__":
    #input_dir = 
    #output_dir = 
    #label_txt = 
    #if "ground" in label_txt:
        #sort_imagenet_1(input_dir, output_dir, label_txt)
    #else:
        #sort_imagenet_2(input_dir, output_dir, label_txt)
    #rename_folders(output_dir)
