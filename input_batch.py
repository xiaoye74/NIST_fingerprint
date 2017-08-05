# -*- coding: utf-8 -*-

import random as rd
import re
import os
import numpy as np
import cv2
#%%

def get_one_label_test(label_file, classes):
    
    f = open(label_file)
    txt = f.read()
    
    pattern = re.compile("W|A|R|L|T")
    match = pattern.search(txt)
    
    def switch_pos(word):
        switcher = {
                    "W":0,
                    "A":1,
                    "R":2,
                    "L":3,
                    "T":4}
        return switcher.get(word,"nothing")
    
    def text2vec(word):
        vector = np.zeros(classes)
        idx = switch_pos(word)
        vector[idx] = 1
        return vector
    
    label = text2vec(match.group())
    
    return label
    
def get_files_test(file_dir, classes=5):
    images = []
    labels = []
    for file in os.listdir(file_dir):
        if file.find(".txt") == -1:
            images.append(file_dir + file)
        else:
            labels.append(get_one_label_test(file_dir+file, classes))
        
    print("there are %d images, %d labels"%(len(images),len(labels)))
    
    return images, labels

def get_one_random_batch(image_list, label_list, batch_size):
    batch_x = np.zeros([batch_size, 512*512])
    batch_y = np.zeros([batch_size, 5])
    def convert2gray(img):
        if len(img.shape) > 2:
            gray = np.mean(img, -1)
            return gray
        else:
            return img
    
    start = rd.randint(0,len(image_list)-batch_size)
    end = start+batch_size
    print(start,end)
    for i in range(start,end):
        image = cv2.imread(image_list[i])
        image = convert2gray(image)
        batch_x[i-start,:] = image.flatten()
        batch_y[i-start,:] = label_list[i]

    return batch_x,batch_y

#%% test
    
    
image_list, label_list = get_files_test("F:\\git_clone_download\\NISTSpecialDatabase4\\NISTSpecialDatabase4GrayScaleImagesofFIGS\\sd04\\png_txt\\figs_0\\")

batch_x,batch_y = get_one_random_batch(image_list,label_list,4)  

