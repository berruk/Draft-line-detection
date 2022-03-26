import pickle
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np


with open("stylegan3-t-ffhq-1024x1024.pkl","rb") as f:
  a = pickle.load(f)

gan = a["G_ema"]

gan.eval()

for param in gan.parameters():
  param.requires_grad = False

begin = torch.randn(1,256)
end = torch.load('pspface.pt')

img = np.zeros((begin.shape[0], begin.shape[1]))
frame_list = []


for i in range(30):

  alpha = i*1.0/30
  morphed = begin(1-alpha) + end*alpha

  img = gan(morphed,0).numpy().squeeze()
  img = np.transpose(img, (1,2,0))
  img[img>1] = 1
  img[img<-1] = -1
  img = 255*(img+1) /2
  newframe = img[:,:,[2,1,0]]

  print("Frame: " +str(i))
  frame_list.append(newframe)

fps = 5
h, w, _ = frame_list[0].shape
video = cv2.VideoWriter('part3.mp4',  0, fps, (w,h))

for image in frame_list:
    video.write(np.uint8(image))

cv2.destroyAllWindows()
video.release()