import numpy as np 
import os
import cv2 as cv 
import moviepy.editor as mpy 

background = cv.imread('Malibu.jpg')

back_h = background.shape[0]
back_w = background.shape[1]

background = cv.resize(background, (int(back_w/back_h * 360), 360))

current_dir = os.getcwd() + '\cat' #cat file directory

img_list = []

for i in range(180):

    img = cv.imread(current_dir + '/cat_' + str(i) + '.png')

    if img is None:
        print("Image can not be opened")

    foreground = np.logical_or(img[:,:,1] < 180, img[:,:,0] > 150) # extract cat from green
    nonzero_x, nonzero_y = np.nonzero(foreground) # extracted cat's pixel coordinates, 1-d array of x and y coords 
    nonzero_cat_values = img[nonzero_x, nonzero_y, :] # 3 channel matrix of cat part pixels of cat
    new_frame = background.copy()
    new_frame[nonzero_x, nonzero_y, :] = nonzero_cat_values #Cat is placed at those exact coords

    frame_w = new_frame.shape[1] - 1 #length in horizontal axis
    mirrored_y = []
    mirrored_x = nonzero_x #same coordinates vertically
    mirrored_y = [frame_w - x for x in nonzero_y] #mirror the coordinats horizontally
    new_frame[mirrored_x, mirrored_y, :] = nonzero_cat_values #mirrored cat is placed

    new_frame = new_frame[:,:,[2,1,0]] #rgb to bgr for moivepy
    img_list.append(new_frame) #add frame to list


clip = mpy.ImageSequenceClip(img_list, fps=25)
audio = mpy.AudioFileClip('selfcontrol_part.wav').set_duration(clip.duration)
clip = clip.set_audio(audioclip=audio)
clip.write_videofile('part1_video.mp4', codec='libx264')
