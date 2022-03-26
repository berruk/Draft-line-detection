
import os
import cv2 
import numpy as np
from scipy.io import loadmat

ground_directory = "C:\\Users\\HP\\Desktop\\hw3\\ground\\test"
network_output_directory = "C:\\Users\\HP\\Desktop\\hw3\\sing_scale_test"

files = os.listdir(network_output_directory)

precision = 0


for i, file in enumerate(files):

    img = cv2.imread(network_output_directory + "\\" + file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, bw_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    img = bw_img

    if img is None:
        continue

    img_name = file.replace('.png', '.mat')

    try:
        mat = loadmat(ground_directory + "\\" + img_name)
    except:
        continue

    map_number = len(mat['groundTruth'][0])
    cum_edges = 0

    for i in range(map_number):
        edges = mat['groundTruth'][0][i][0][0][1] #edges
        norm_edges = edges * 255
        cum_edges += norm_edges

    if img.shape == cum_edges.shape:
        tp = np.sum(np.logical_and(img == 255, cum_edges == 255))
        fp = np.sum(np.logical_and(img == 255, cum_edges == 0))

        if tp + fp != 0:
            precision += tp / (tp + fp)
    else:
        print("Non matching inputs") 
        
print(precision/200)



