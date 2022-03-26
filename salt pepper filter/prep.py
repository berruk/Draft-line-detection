import moviepy.video.io.VideoFileClip as mpy
import moviepy
import cv2 
import numpy as np


def median_filter(img, kernel_size = 3.5):
 
    final = np.zeros_like(img)
    border = 1
    median_index = int(kernel_size*kernel_size/2)

    for i in range(border, img.shape[0] - border):
        for j in range(border, img.shape[1] - border):
            kernel = (img[i-border : i+border + 1, j-border : j+border+1]).flatten()
            temp = np.sort(kernel)
            final[i, j] = temp[median_index]

    return final


def main():

    vid = mpy.VideoFileClip('shapes_video.mp4')
    frame_count = vid.reader.nframes
    video_fps = vid.fps
    new_video = []
    for i in range(frame_count):
        frame = vid.get_frame(i*1.0/video_fps)
        filtered = median_filter(frame,4)
        new_frame = filtered[:,:,[2,1,0]] #rgb to bgr for moivepy
        #cv2.imwrite(str(i) + '.jpg',new_frame)
        new_video.append(new_frame)

        print(i)

    h = new_video[0].shape[0]
    w = new_video[0].shape[1]
    
    video = cv2.VideoWriter('part1.mp4',  0, video_fps, (w,h))
    for image in new_video:
        video.write(np.uint8(image))

    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    main()