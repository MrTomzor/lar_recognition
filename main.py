import cv2
#import imageio
import numpy as np
from scipy.io import loadmat
from scipy.misc import imsave

RED = 0
GREEN = 55
BLUE = 120 
POLE_RADIUS = 25

data = loadmat('2020-04-23-10-15-56.mat')
img = data['image_rgb']
imsave('test_img.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
print(data['K_rgb'])  
K = data['K_rgb']

def drawRect(img_hsv, x, y, w, h):
    clr = [0, 255, 255]
    for xx in range(w):
        img_hsv[y][x + xx] = clr
        img_hsv[y + h][x + xx] = clr
    for yy in range(h):
        img_hsv[y + yy][x] = clr
        img_hsv[y + yy][x + w] = clr
    return img_hsv

def getUnitVector(vector):
    return vector / np.linalg.norm(vector)

def getVectorsAngle(v1, v2):
    return np.arccos(np.dot(np.transpose(v1),v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def getContours(color, img_hsv):
    hueThreshold = 10
    saturationThreshold = 10
    valueThreshold = 10

    mask = cv2.inRange(img_hsv, np.array([color - hueThreshold, 50 if (color > 40 and color < 60) else 150, 50]) , np.array([color + hueThreshold, 255, 255]))

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    
    imsave('test_mask.png',[[[x, x, x] for x in row] for row in mask]) 
    return contours

def getPoles(colors, img_hsv, img_d, K):
    poles = []
    invK = np.linalg.inv(K)
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
                spaceVector = np.dot(invK, np.array([cx, cy, 1]))
                spaceVector[1] = 0 # The y coordinate would only mess with distances
                spaceVector = spaceVector * img_d[cy,cx]
                if img_d[cy, cx] != 0:
                    pole = {'color': color, 'col': cx, 'row': cy, 'depth': img_d[cy, cx], 'bound': [x, y, w, h], 'spaceVector': spaceVector} 
                    print(pole)
                    poles.append(pole)
    return poles

def getGateParams(leftPole, rightPole):
    # Transform the coordinates of poles to a top down representation where z -> y, x->x
    lPos = np.array([leftPole['spaceVector'][0], leftPole['spaceVector'][2]])
    rPos = np.array([rightPole['spaceVector'][0], rightPole['spaceVector'][2]])
    gateWidth = np.linalg.norm(rPos - lPos) - 2 * POLE_RADIUS
    gateCenterPos = (lPos + rPos) / 2
    gateFacingDir = rPos - lPos
    gateFacingDir = np.array([gateFacingDir[1], gateFacingDir[0]])
    robotFacingDir = np.array([0,1])
    print('angle between poles', getVectorsAngle(lPos, rPos) * 180 / 3.14159267)
    print('alpha', getVectorsAngle(robotFacingDir, gateFacingDir) * 180 / 3.14159267)
    print('beta', getVectorsAngle(robotFacingDir, gateCenterPos) * 180 / 3.14159267)
    print('distance from gate\'s center', np.linalg.norm(gateCenterPos))
    print('goal width:', gateWidth) 
    return 0

def solveProblem2(img_hsv):
    # Pick nearest two green poles and set the one on the left as leftPole 
    # and the other one as rightPole
    poles = getPoles([GREEN], img_hsv, data['image_depth'], K) 
    if len(poles) < 2:
        print('ERROR: Fewer than 2 green poles detected!')
        exit(100)
    poles = sorted(poles, key=lambda x: x.get('depth'), reverse = True)
    if poles[0]['col'] < poles[1]['col']:
        leftPole = poles[0]
        rightPole = poles[1]
    else:
        leftPole = poles[1]
        rightPole = poles[0]
    getGateParams(leftPole, rightPole)


#imsave('test_rect.png',drawRect(img, 30, 30, 100, 100))
#print(np.dot(K,np.array([[0],[0],[100]])))
#print(np.dot(np.linalg.inv(K),[[600],[233],[1]]))
solveProblem2(cv2.cvtColor(data['image_rgb'], cv2.COLOR_BGR2HSV))
