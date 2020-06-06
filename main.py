import cv2
#import imageio
import numpy as np
from scipy.io import loadmat
from scipy.misc import imsave

RED = 0
GREEN = 55
BLUE = 120 
data = loadmat('2020-04-23-10-15-56.mat')
img = data['image_rgb']
imsave('test_img.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
print(data['K_rgb'])  

def drawRect(img_hsv, x, y, w, h):
    clr = [0, 255, 255]
    for xx in range(w):
        img_hsv[y][x + xx] = clr
        img_hsv[y + h][x + xx] = clr
    for yy in range(h):
        img_hsv[y + yy][x] = clr
        img_hsv[y + yy][x + w] = clr
    return img_hsv

def getVectorsAngle():
    return 0

def getContours(color, img_hsv):
    hueThreshold = 10
    saturationThreshold = 10
    valueThreshold = 10

    mask = cv2.inRange(img_hsv, np.array([color - hueThreshold, 50 if (color > 40 and color < 60) else 150, 50]) , np.array([color + hueThreshold, 255, 255]))

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    
    imsave('test_mask.png',[[[x, x, x] for x in row] for row in mask]) 
    return contours

def getAllPoles(colors, img_hsv, img_d, K):
    poles = []
    for color in colors:
        contours = getContours(color, img_hsv)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            lng = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)
            # print(h/w, area)
            if ((h/w) > 3.5) and (area > 1000):
                M = cv2.moments(cnt)
                cx = int(round(M['m10'] / M['m00']))
                cy = int(round(M['m01'] / M['m00']))
                print('center', cy, cx)
                spaceVector = None
                if img_d[cy, cx] != 0:
                    pole = {'color': color, 'col': cx, 'row': cy, 'depth': img_d[cy, cx], 'bound': [x, y, w, h], 'spaceVector': spaceVector} 
                    print(pole)
                    poles.append(pole)

def getGoalParams():
    return 0

def solveProblem1():
    return 0

imsave('test_rect.png',drawRect(img, 30, 30, 100, 100))
getAllPoles([GREEN], cv2.cvtColor(data['image_rgb'], cv2.COLOR_BGR2HSV), data['image_depth'], data['K_rgb'])
