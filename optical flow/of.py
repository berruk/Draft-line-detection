import numpy as np
import cv2 
from os import listdir
import moviepy.editor as moviepy
import math


new_video = []
def BS(prev, next):
    diff = cv2.absdiff(prev, next)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    kernel_3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (4,4))
    closing = cv2.morphologyEx(diff, cv2.MORPH_ERODE, kernel_3)
    closing = cv2.morphologyEx(diff, cv2.MORPH_DILATE, kernel)
    
    (T, thresh) = cv2.threshold(closing, 5, 255,
	cv2.THRESH_BINARY)

    #cv2.waitKey()

    return closing


def OF(prev, next, result, i):

    #cv2.imshow("prev",prev)
    #cv2.imshow("next",next)

    resw = next.shape[0]
    resh = next.shape[1]

    features = cv2.goodFeaturesToTrack(prev, 500, 0.07, 15)

    next = next/255
    prev = prev/255

    u = np.zeros(prev.shape)
    v = np.zeros(prev.shape)

    #Gradient of x y t
    dx = cv2.filter2D(prev, -1, np.array([[-1, 1], [-1, 1]]))              
    dy = cv2.filter2D(prev, -1, np.array([[-1, -1], [1, 1]]))              
    t =  cv2.filter2D(next, -1, np.ones((2,2))) 
    dt = cv2.filter2D(prev, -1, np.ones((2,2))) 

    window = 5

    for feature in features:

        x = int(feature[0][1])
        y = int(feature[0][0])		
        
        Ix = dx[x - window : x + window, y : y + window]
        Iy = dy[x - window: x + window, y : y + window]
        dt = t - dt
        It = dt[x - window: x + window, y : y + window]

        A = np.vstack((Ix.flatten(), Iy.flatten()))
        b = It.flatten()
        b = np.reshape(b, (b.shape[0],1))
        res = np.matmul(np.linalg.pinv(A.T), b)     

        u[x,y] = res[0][0]
        v[x,y] = res[1][0]

    tempres = result.copy()
    count = 0
    if True:
        for i in range(resw):
            for j in range(resh):
                if u[i][j] and v[i][j]:
                    u_current, v_current = u[i][j], v[i][j]
                    count += 1
                    result = cv2.arrowedLine(result, (i, j), (int(j+v_current), int(i+u_current)),
                                            (255, 0, 0), thickness = 3)

    if count > 5:
        return tempres   
    return result
    

files = listdir("DJI_0101")
step = 1
for i in range(step, len(files)-1,step):

    img0 = cv2.imread("DJI_0101/" + files[i-step])
    img1 = cv2.imread("DJI_0101/" + files[i])
    img2 = cv2.imread("DJI_0101/" + files[i+step])

    result = img2

    prev = BS(img0,img1)
    next = BS(img1,img0)

    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)


    result = OF(prev, next, result, i)
    #cv2.imshow("res",result)
    #cv2.waitKey()
    #cv2.imwrite("result/" + str(i) +".png", result)

    new_video.append(result[:, :, [2,1,0]])

    print(i)


clip = moviepy.ImageSequenceClip(new_video, fps = 25)
clip.write_videofile("of.mp4", codec="libx264")

