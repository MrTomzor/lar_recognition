import cv2
import sys
import numpy as np
from scipy.io import loadmat
from scipy.misc import imsave

RED = 0 
GREEN = 62
BLUE = 95 
COLOR_NAMES = {RED: 'RED', GREEN: 'GREEN', BLUE: 'BLUE'}
POLE_RADIUS = 25
ROBOT_WIDTH = 400
SAFE_ROBOT_PASS_DISTANCE = 5


def drawRect(img_hsv, x, y, w, h):
    clr = [0, 255, 255]
    for xx in range(w):
        img_hsv[y][x + xx] = clr
        img_hsv[y + h][x + xx] = clr
    for yy in range(h):
        img_hsv[y + yy][x] = clr
        img_hsv[y + yy][x + w] = clr
    return img_hsv

def getVectorsAngle(v1, v2):
    return np.arccos(np.dot(np.transpose(v1),v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def getContours(color, img_hsv):
    hueThreshold = 20
    mask = cv2.inRange(img_hsv, np.array([color - hueThreshold, 70 if color == GREEN else 150, 50]) , np.array([color + hueThreshold, 255, 255]))

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    
    imsave('mask_' + COLOR_NAMES[color] + '.png',[[[x, x, x] for x in row] for row in mask]) 
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
            #print('AREA',  area)
            if (area > 400):
                M = cv2.moments(cnt)
                cx = int(round(M['m10'] / M['m00']))
                cy = int(round(M['m01'] / M['m00']))
                #print('center', cy, cx)
                spaceVector = np.dot(invK, np.array([cx, cy, 1]))
                spaceVector[1] = 0 # The y coordinate would only mess with distances
                spaceVector = spaceVector * img_d[cy,cx]
                if img_d[cy, cx]!= 0:
                    pole = {'color': color, 'col': cx, 'row': cy, 'depth': img_d[cy, cx], 'bound': [x, y, w, h], 'spaceVector': spaceVector}
                    print(pole)
                    poles.append(pole)
    return poles

def getGateParameters(leftPole, rightPole, imageWidth):
    # Transform the coordinates of poles to a top down representation where z -> y, x->x
    lPos = np.array([leftPole['spaceVector'][0], leftPole['spaceVector'][2]])
    rPos = np.array([rightPole['spaceVector'][0], rightPole['spaceVector'][2]])

    gateImageWidth = rightPole['bound'][0] - (leftPole['bound'][0] + leftPole['bound'][2])
    gateImageCenter = (rightPole['col'] + leftPole['col']) / 2 - imageWidth / 2

    gateCenterPos = (lPos + rPos) / 2
    gateFacingDir = rPos - lPos
    gateFacingDir = np.array([gateFacingDir[1], gateFacingDir[0]])
    robotFacingDir = np.array([0,1])
    
    alpha = getVectorsAngle(robotFacingDir, gateFacingDir)
    beta = getVectorsAngle(robotFacingDir, gateCenterPos)
    robotDist = np.linalg.norm(gateCenterPos) 
    gateWidth = np.linalg.norm(rPos - lPos) - 2 * POLE_RADIUS
    visibleWidth = np.cos(alpha) * gateWidth
    robotWillPassAfterRotation = visibleWidth > ROBOT_WIDTH + 2 * SAFE_ROBOT_PASS_DISTANCE
    return {'alpha': alpha, 'beta': beta, 'v': robotDist, 'd': gateWidth, 'willPass': robotWillPassAfterRotation, 'c': gateImageCenter, 'g': gateImageWidth}

def getNearestGateParameters(data):
    gatesDetected = []
    img_hsv = cv2.cvtColor(data['image_rgb'], cv2.COLOR_BGR2HSV) 
    img_depth = data['image_depth']
    K = data['K_rgb']

    for color in [GREEN, RED, BLUE]:
        poles = getPoles([color], img_hsv, img_depth, K)
        if len(poles) < 2:
            print('INFO: Fewer than 2 poles of color ' + COLOR_NAMES[color] +  ' detected!')
            continue
        poles = sorted(poles, key=lambda x: x.get('depth'), reverse = True)
        if poles[0]['col'] < poles[1]['col']:
            leftPole = poles[0]
            rightPole = poles[1]
        else:
            leftPole = poles[1]
            rightPole = poles[0]
        params = getGateParameters(leftPole, rightPole, len(img_hsv[0]))
        params['color'] = COLOR_NAMES[color]
        #print('INFO: Nearest gate of color ' + COLOR_NAMES[color] + ': ',  params)
        gatesDetected.append(params)
    if len(gatesDetected) == 0:
        print('ERROR: Fewer than 2 poles found for each color, no gate detected')
        exit(100)
    gatesDetected = sorted(gatesDetected, key=lambda x: x.get('v'))
    nearestGate = gatesDetected[0]
    #print('INFO: Nearest gate found :',  gatesDetected[0])
    print('Color of gate: ', nearestGate['color'])
    print('Goal center distance from center of image', nearestGate['c'])
    print('Goal width from image', nearestGate['g'])
    print('Alpha', nearestGate['alpha'])
    print('Beta', nearestGate['beta'])
    print('Distance from gate\'s center', nearestGate['v'])
    print('Gate width:', nearestGate['d']) 
    print('Will robot pass after rotation?', nearestGate['willPass'])
    return 

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('ERROR: No filename given! Please add the data filename as an argument while calling this script')
        exit(101)
    data = loadmat(sys.argv[1])
    imsave('testimg.png',cv2.cvtColor(data['image_rgb'], cv2.COLOR_BGR2RGB))
    getNearestGateParameters(data)
