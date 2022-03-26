import numpy as np
import cv2
import os
import moviepy.editor as moviepy
import matplotlib.pyplot as plt 
from scipy.spatial import distance 


# opencv doc comment
   #
   #            a                         x    b
   #/ x0 y0  1  0  0  0 -x0*u0 -y0*u0 \ /c00\ /u0\
   #| x1 y1  1  0  0  0 -x1*u1 -y1*u1 | |c01| |u1|
   #| x2 y2  1  0  0  0 -x2*u2 -y2*u2 | |c02| |u2|
   #| x3 y3  1  0  0  0 -x3*u3 -y3*u3 |.|c10|=|u3|
   #|  0  0  0 x0 y0  1 -x0*v0 -y0*v0 | |c11| |v0|
   #|  0  0  0 x1 y1  1 -x1*v1 -y1*v1 | |c12| |v1|
   #|  0  0  0 x2 y2  1 -x2*v2 -y2*v2 | |c20| |v2|
   #\  0  0  0 x3 y3  1 -x3*v3 -y3*v3 / \c21/ \v3/
   #
 

def euclid(a, b):
	return int(np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2))

def PerspectiveTransform(src, dst):

    a = np.zeros((8, 8))
    b = np.zeros((8))

    a[0:4, 3:6] = 0
    a[0:4, 2] = 1
    a[4:8, 0:3] = 0
    a[4:8, 5] = 1

    for i in range(4):
        a[i][0]   =  src[i][0]
        a[i][1]   =  src[i][1]
        a[i][6]   = -src[i][0] * dst[i][0]
        a[i][7]   = -src[i][1] * dst[i][0]
        b[i] = dst[i][0]

    for i in range(4,8):   
        j = i-4 
        a[i][6] = -src[j][0] * dst[j][1]
        a[i][7] = -src[j][1] * dst[j][1]
        b[i]    =  dst[j][1]
        a[i][4] =  src[j][1]
        a[i][3] =  src[j][0]

    a_inv = np.linalg.pinv(a)
    x = np.dot(a_inv, b)
    
    return np.append(x, 1).reshape((3,3))

def ordered_points(pts):

	x_sorted_indexes = np.argsort(pts[:, 0])
	xcoord = pts[x_sorted_indexes, :]
	
	right_half = np.zeros((2,2))
	right_half[0,:] = xcoord[2,:]
	right_half[1,:] = xcoord[3,:]

	left_half = np.zeros((2,2))
	left_half[0,:] = xcoord[0,:]
	left_half[1,:] = xcoord[1,:]

	if left_half[0,1] > left_half[1,1]:
		x1 = left_half[1,:]
		x3 = left_half[0,:]

	else:	
		x1 = left_half[0,:]
		x3 = left_half[1,:]

	euclidean = distance.cdist(x1[None], right_half, "euclidean")
	(x4, x2) = right_half[np.argsort(euclidean[0])[::-1], :]

	return np.array([x1, x2, x3, x4], 
					dtype="float32")	

def main():

	planes = np.zeros((9,472,4,3))

	for i in range(1,10):
		with open("Plane_" + str(i) + ".txt") as f:
			content = f.readlines()
			for line_id in range(len(content)):
				sel_line = content[line_id]
				sel_line = sel_line.replace(')\n', '').replace("(", '').split(")")

				for point_id in range(4):
					sel_point = sel_line[point_id].split(" ")

					planes[i-1,line_id,point_id,0] = float(sel_point[0])
					planes[i-1,line_id,point_id,1] = float(sel_point[1])
					planes[i-1,line_id,point_id,2] = float(sel_point[2])

	images_list = []

	cat_orig = cv2.imread('cat-headphones.png')
	cover = cv2.imread('cover.jpg')

	#extract and resize cat
	cat = cv2.resize(cat_orig, (250,250), interpolation = cv2.INTER_AREA)
	gray = cv2.cvtColor(cat, cv2.COLOR_RGB2GRAY)
	cat_x, cat_y = np.nonzero(gray) 
	cat_values = cat[cat_x, cat_y, :] 
	cat_x += 70
	cat_y += 140


	for i in range(472):

		#create white background
		blank_image = np.zeros((322,572,3), np.uint8)
		blank_image[:] = (255, 255, 255)
		blank_image[cat_x, cat_y, :] = cat_values

		#empty lists for front images
		front_nonzero_x = []
		front_nonzero_y = []
		front_pixels = []

		for j in range(9):

			pts   = planes[j,i,:,:].squeeze()[:,0:2].astype(np.int32)	
			depth = planes[j,i,:,2][0]

			temp = np.copy(pts[3,:])
			pts[3, :] = pts[2,:]
			pts[2, :] = temp

			pts = pts.reshape((-1, 1, 2))
			img = cv2.imread('cover.jpg')
			rows, cols, ch = img.shape

			#order destination coordinates 
			pt = ordered_points(pts[:,0])

			#find longest vertices
			min_row = int(min(pt[2][1], pt[0][1], pt[3][1], pt[1][1]))
			max_row = int(max(pt[2][1], pt[0][1], pt[3][1], pt[1][1]))
			max_col = int(max(pt[2][0], pt[0][0], pt[3][0], pt[1][0]))
			min_col = int(min(pt[2][0], pt[0][0], pt[3][0], pt[1][0]))

			if (max_row >= min_row + 1) and (max_col >= min_col + 1):

				#create array with desired coordinates, normalized
				dst = np.array([[pt[1][0]-min_col, pt[1][1]-min_row], 
							    [pt[0][0]-min_col, pt[0][1]-min_row],
								[pt[2][0]-min_col, pt[2][1]-min_row], 
								[pt[3][0]-min_col, pt[3][1]-min_row]], 
								dtype='float32')


				#create array with input images coordinates
				src = np.array([[0,0], [img.shape[0], 0], 
								[img.shape[0],img.shape[1]], 
								[0,img.shape[1]],], 
								dtype='float32')

				#get transform matrix and warp image
				perspectiveTransform = PerspectiveTransform(src, dst)
				warped = cv2.warpPerspective(img, perspectiveTransform, (max_col - min_col, max_row - min_row))

				#get contours of warped image
				clone_warped = warped.copy()
				clone_warped = cv2.cvtColor(clone_warped, cv2.COLOR_BGR2GRAY)
				contours, _ = cv2.findContours(clone_warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

				#create a mask
				mask = np.zeros((clone_warped.shape[0], clone_warped.shape[1]))
				cv2.drawContours(mask, contours, 0, (255, 255, 255), -1)

				#extract the image
				nonzero_x, nonzero_y = np.nonzero(mask) 
				nonzero_pixels = warped[nonzero_x, nonzero_y, :] 
				nonzero_x += min_row 
				nonzero_y += min_col 

				#if plane is at front, keep to write later
				if  depth < 245 :
					front_nonzero_x.append(nonzero_x)
					front_nonzero_y.append(nonzero_y)
					front_pixels.append(nonzero_pixels)

				#if back, write
				else: #add to frame
					blank_image[nonzero_x, nonzero_y, :] = nonzero_pixels

				#cv2.imshow("h",blank_image)
				#cv2.waitKey(0)

		#backgrodun is laid, add cat
		blank_image[cat_x, cat_y, :] = cat_values

		#add foreground planes
		for i in range(len(front_nonzero_x)):
			blank_image[front_nonzero_x[i], 
			front_nonzero_y[i], :] = front_pixels[i]

		#bgr 
		blank_image = blank_image[:, :, [2,1,0]]
		images_list.append(blank_image)

		print("Frame: " + str(i+1))

	clip = moviepy.ImageSequenceClip(images_list, fps = 25)
	clip.write_videofile("part3_video.mp4", codec="libx264")

if __name__ == "__main__":
    main()