import numpy as np 
import os
import cv2 as cv 
import moviepy.editor as mpy 


def save_clip(img_list):

    clip = mpy.ImageSequenceClip(img_list, fps=25)
    audio = mpy.AudioFileClip('selfcontrol_part.wav').set_duration(clip.duration)
    clip = clip.set_audio(audioclip=audio)
    clip.write_videofile('part2_video.mp4', codec='libx264')

def match_hist(cat_hist, image, target_img):
    
    for k in range(3):

        bns = 255

        #cat_hist, _ = np.histogram(image[:,:,k].flatten(), bns, density=True)
        target_hist, bins = np.histogram(target_img[:,:,k].flatten(), bns, density=True)

        cdf_cat = bns * np.cumsum(cat_hist)
        cdf_target = bns * np.cumsum(target_hist) 
 
        cdf_cat =  cdf_cat / cdf_cat[-1]
        cdf_target = cdf_target / cdf_target[-1]

        temp = np.interp(np.interp(image[:,:,k].flatten(), bins[:-1], cdf_cat), cdf_target, bins[:-1])
        image[:,:,k] = temp.reshape((image.shape[0], image.shape[1]))

    return image 

def masked_image(num_frames):

    hist = np.zeros((3,256,256))

    # each frame's directory
    current_dir = os.getcwd() + '\cat' 
    img = cv.imread(current_dir + '/cat_' + str(num_frames) + '.png')

    if img is None:
        print("Image can not be opened")

    foreground = np.logical_or(img[:,:,1] < 180, img[:,:,0] > 150) # extract cat from green

    # create a mask to obtain cat parts' histogram 
    mask = np.empty((img.shape[0], img.shape[1]))
    mask.fill(False)
    mask[:, 0:foreground.shape[1]] = foreground
    mask = mask.astype(np.uint8)

    masked = np.zeros((360,640,3), np.uint8)
    for i in range(3):
        masked[:,:,i] = cv.bitwise_and(img[:,:,i], img[:,:,i], mask = mask)

    return masked

def cal_avg_hist(num_frames):
    masked_cat = masked_image(0)
    bns = 255
    for k in range(3):
        cat_hist, _ = np.histogram(masked_cat[:,:,k].flatten(), bns, density=True)

    for i in range(1,num_frames):
        for k in range(3):
            masked_cat = masked_image(i)
            hist, _ = np.histogram(masked_cat[:,:,k].flatten(), bns, density=True)
            cat_hist += hist

    return cat_hist/num_frames   


def main():

    img_list = []
    num_frames = 180    
    target_img = cv.imread("monet.jpg")
    if target_img is None:
            print("Background image can not be opened")

    background = cv.imread('Malibu.jpg')
    if background is None:
            print("Background image can not be opened")
       
    avg_cat_hist = cal_avg_hist(num_frames)

    for i in range(num_frames):

        masked_cat = masked_image(i)

        back_h = background.shape[0]
        back_w = background.shape[1]
        background = cv.resize(background, (int(back_w/back_h * 360), 360))

        # each frame's directory
        current_dir = os.getcwd() + '\cat' 
        img = cv.imread(current_dir + '/cat_' + str(i) + '.png')

        if img is None:
            print("Image can not be opened")

        #Adding original cat
        foreground = np.logical_or(img[:,:,1] < 180, img[:,:,0] > 150) # extract cat from green
        nonzero_x, nonzero_y = np.nonzero(foreground) # extracted cat's pixel coordinates, 1-d array of x and y coords 
        nonzero_cat_values = img[nonzero_x, nonzero_y, :] # 3 channel matrix of cat part pixels of cat
        new_frame = background.copy()
        new_frame[nonzero_x, nonzero_y, :] = nonzero_cat_values #Cat is placed at those exact coords

        #Adding matched cat
        matched_cat = match_hist(avg_cat_hist, masked_cat, target_img)
        matched_cat_values = matched_cat[nonzero_x, nonzero_y, :]
        frame_w = new_frame.shape[1] - 1 #length in horizontal axis
        mirrored_y = []
        mirrored_x = nonzero_x #same coordinates vertically
        mirrored_y = [frame_w - x for x in nonzero_y] #mirror the coordinats horizontally
        new_frame[mirrored_x, mirrored_y, :] = matched_cat_values #mirrored cat is placed

        new_frame = new_frame[:,:,[2,1,0]] #rgb to bgr for movie
        img_list.append(new_frame) #add frame to list


    save_clip(img_list)


if __name__ == "__main__":
    main()