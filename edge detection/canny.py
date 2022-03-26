import os
import cv2 
import numpy as np
from scipy.io import loadmat

ground_directory = "C:\\Users\\HP\\Desktop\\hw3\\ground\\test"
img_directory = "C:\\Users\\HP\\Desktop\\hw3\\images\\test"
network_output_directory = "C:\\Users\\HP\\Desktop\\hw3\\sing_scale_test"

files = os.listdir(ground_directory)

precision = 0


for i, file in enumerate(files):

    try:
        mat = loadmat(ground_directory + "\\" + file)
    except:
        continue
    
    img_name = file.replace('.mat', '.jpg')

    img = cv2.imread(img_directory + "\\" + img_name)

    if img is None:
        continue

    map_number = len(mat['groundTruth'][0])
    cum_edges = 0

    #adaptive threshold is somehow worse
    #blurred_img = cv2.blur(img, ksize=(5,5))
    #med_val = np.median(img) 
    #lower = int(max(0 ,1*med_val))
    #upper = int(min(255,1.6*med_val))

    for i in range(map_number):
        edges = mat['groundTruth'][0][i][0][0][1] #edges
        norm_edges = edges * 255
        cum_edges += norm_edges

    canny_edges = cv2.Canny(img, 100, 800)

    cv2.imwrite("C:\\Users\\HP\\Desktop\\hw3\\edges\\" + img_name, canny_edges)


    #cv2.imshow("canny", canny_edges)
    #cv2.imshow("cum", cum_edges)
    #cv2.waitKey(0)

    tp = np.sum(np.logical_and(canny_edges == 255, cum_edges == 255))
    fp = np.sum(np.logical_and(canny_edges == 255, cum_edges == 0))
    
    if tp + fp != 0:
        precision += tp / (tp + fp)

print(precision/200)



