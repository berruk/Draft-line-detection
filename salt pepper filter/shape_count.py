import moviepy.video.io.VideoFileClip as mpy
import moviepy
import cv2 
import numpy as np

#Returns the new card in every frame
def get_card(frame1, frame2):

    #get 2 consecutive frames
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    (_, bin1) = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
    (_, bin2) = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)

    #remove prev
    sub = bin2 - bin1

    #obtain a mask for the fresh card
    opening = cv2.morphologyEx(sub, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)))
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(100,100)))

    masked = cv2.bitwise_and(bin2, closing)
    cv2.imshow("w", masked)
    cv2.waitKey(0)
    return masked

def main():

    vid = mpy.VideoFileClip('part1.mp4')
    video_fps = vid.fps

    count = np.zeros((3,1))

    for i in range(6,80): #only counting the frames with shapes 

        frame1 = vid.get_frame(i*1.0/video_fps)
        frame2 = vid.get_frame((i+1)*1.0/video_fps)

        card = get_card(frame1, frame2)

        #find contours
        contours, _ = cv2.findContours(card, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        new_img = np.zeros_like(card)
        new_img.fill(255)
        
        shapes = []

        #detect shape
        for contour in contours:
            
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            vertices = len(approx)
        
            cv2.drawContours(new_img, contours, -1, (0, 0, 0), 1)

            if vertices == 4 :
                _, _ , w, h = cv2.boundingRect(approx)
                aspectRatio = float(w)/h
        
                if aspectRatio >= 0.90 and aspectRatio < 1.10:
                    shapes.append("square")
        
            elif vertices == 5:
                shapes.append("pentagon")
        
            elif vertices == 10 :
                shapes.append("star")

            elif vertices == 6:
                shapes.append("pentagon")   

        shapes = list(dict.fromkeys(shapes))
        
        shape = ""
        if len(shapes) == 1:
            shape = shapes[0]
            
        elif len(shapes) > 1:
            if "square" in shapes:
                shapes.remove("square")     
            shape = shapes[0]
        
        if shape == "star":
            count[0] += 1
        elif shape == "pentagon":
            count[1] += 1
        elif shape == "square":
            count[2] += 1

        print(shape)
        #cv2.imshow('labeled.jpg', new_img)
        #cv2.waitKey(0)

    print("star" + str(count[0]))    
    print("pentagon: " + str(count[1]))    
    print("square: " + str(count[2]))    

if __name__ == "__main__":
    main()