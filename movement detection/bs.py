import numpy as np
import cv2 
from os import listdir
import moviepy.editor as moviepy

files = listdir("DJI_0101")

new_video = []

for i in range(4, len(files)-1,):

    background = cv2.imread("DJI_0101/" + files[i-4])
    image = cv2.imread("DJI_0101/" + files[i])
    diff = cv2.absdiff(background, image)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2))
    opening = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    new_video.append(closing)
    print(i)
    #cv2.imshow('image',closing)
    #cv2.waitKey()

clip = moviepy.ImageSequenceClip(new_video, fps = 25)
clip.write_videofile("blobs2.mp4", codec="libx264")

