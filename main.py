import cv2
#import imageio
import numpy as np
from scipy.io import loadmat
from scipy.misc import imsave

data = loadmat('2020-04-23-10-15-56.mat')
img = data['image_rgb']
print(len(img))
imsave('test_img.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  
def getVectorsAngle():
    return 0

def getContours(color, img_hsv):
    hueThreshold = 10
    saturationThreshold = 10
    valueThreshold = 10
    width = len(img_hsv[0])
    height = len(img_hsv)
    print("img width", width, "img height", height)

    mask = cv2.inRange(img_hsv, np.array([color - hueThreshold, 50 if (color > 40 and color < 60) else 150, 50]) , np.array([color + hueThreshold, 255, 255]))
    print(mask)

    #imageio.imwrite("mask.png", mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    
    imsave('test_mask.png',[[[x, x, x] for x in row] for row in mask]) 

def getAllPoles():
    return 0

def getGoalParams():
    return 0

def solveProblem1():
    return 0

getContours(50, cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
