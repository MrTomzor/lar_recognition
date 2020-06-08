import cv2
import sys
import numpy as np
from scipy.io import loadmat
from scipy.misc import imsave
import math

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
    if np.cross(v1, v2) > 0:
        return np.arccos(np.dot(np.transpose(v1),v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return -np.arccos(np.dot(np.transpose(v1),v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def getContours(color, img_hsv):
    hueThreshold = 20
    mask = cv2.inRange(img_hsv, np.array([color - hueThreshold, 70 if color == GREEN else 150, 50]) , np.array([color + hueThreshold, 255, 255]))

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    
    #imsave('mask_' + COLOR_NAMES[color] + '.png',[[[x, x, x] for x in row] for row in mask]) 
    return contours

def getPoleCorners(pole):
    tmp = []
    for i in range(4):
        if pole['rect'][i][1] > pole['row']:
            tmp.append(pole['rect'][i])

    if tmp[0][0] < tmp[1][0]:
        left_corner = tmp[0]
        right_corner = tmp[1]
    else:
        left_corner = tmp[1]
        right_corner = tmp[0]
    return [left_corner, right_corner]

def getPoles(colors, img_hsv, img_d, K, pc):
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
                rect = cv2.minAreaRect(cnt)
                #spaceVector = np.dot(invK, np.array([cx, cy, 1]))
                #spaceVector[1] = 0 # The y coordinate would only mess with distances
                #spaceVector = spaceVector * img_d[cy,cx]
                spaceVector = pc[cy][cx] * 1000
                if img_d[cy, cx]!= 0:
                    pole = {'color': color, 'col': cx, 'row': cy, 'dist2': (spaceVector[0]**2 + spaceVector[2]**2), 'bound': [x, y, w, h], 'spaceVector': spaceVector, 'rect': cv2.boxPoints(rect)}
                    #print('spaceVector = ', spaceVector)
                    poles.append(pole)
    return poles

def getGateParameters(leftPole, rightPole, imageWidth):
    # Transform the coordinates of poles to a top down representation where z -> y, x->x
    lPos = np.array([leftPole['spaceVector'][0], leftPole['spaceVector'][2]])
    rPos = np.array([rightPole['spaceVector'][0], rightPole['spaceVector'][2]])

    gateLeftCornerCol = getPoleCorners(leftPole)[1][0]
    gateRightCornerCol = getPoleCorners(rightPole)[0][0]

    gateImageWidth = gateRightCornerCol - gateLeftCornerCol 
    gateImageCenter = (gateRightCornerCol + gateLeftCornerCol) / 2 - imageWidth / 2

    gateCenterPos = (lPos + rPos) / 2
    gateFacingDir = rPos - lPos
    gateFacingDir = np.array([-gateFacingDir[1], gateFacingDir[0]])
    robotFacingDir = np.array([0,1])
    
    beta =  getVectorsAngle(gateCenterPos, gateFacingDir) #* 180 / 3.14159267 
    alpha = getVectorsAngle(robotFacingDir, gateCenterPos)# * 180 / 3.14159267 
    robotDist = np.linalg.norm(gateCenterPos) 
    gateWidth = np.linalg.norm(rPos - lPos) - 2 * POLE_RADIUS
    if gateWidth > 600 or gateWidth < 400:
        return None
    visibleWidth = np.cos(alpha) * gateWidth
    robotWillPassAfterRotation = visibleWidth > ROBOT_WIDTH + 2 * SAFE_ROBOT_PASS_DISTANCE
    return {'alpha': alpha, 'beta': beta, 'v': robotDist, 'd': gateWidth, 'willPass': robotWillPassAfterRotation, 'c': gateImageCenter, 'g': gateImageWidth}

def getNearestGateParameters(data):
    gatesDetected = []
    img_hsv = cv2.cvtColor(data['image_rgb'], cv2.COLOR_BGR2HSV) 
    img_depth = data['image_depth']
    K = data['K_rgb']
    pc = data['point_cloud']

    for color in [GREEN, RED, BLUE]:
        # Get poles of this color from the image and sort them by depth
        poles = getPoles([color], img_hsv, img_depth, K, pc)
        if len(poles) < 2:
            #print('INFO: Fewer than 2 poles of color ' + COLOR_NAMES[color] +  ' detected!')
            continue
        poles = sorted(poles, key=lambda x: x.get('dist2'),reverse=False)
        if poles[0]['col'] < poles[1]['col']:
            leftPole = poles[0]
            rightPole = poles[1]
        else:
            leftPole = poles[1]
            rightPole = poles[0]
        # Calculate gate parameters. If calculating from the two nearest poles gives weird results,
        # then the robot is probably inside a gate and the nearest pole is discarded
        params = getGateParameters(leftPole, rightPole, len(img_hsv[0]))
        if params == None:
            print("INFO: Nearest " + COLOR_NAMES[color] + " gate too wide or too narrow, robot may be inside the gate")
            if len(poles) < 3:
                continue
            poles.pop(0)
            if poles[0]['col'] < poles[1]['col']:
                leftPole = poles[0]
                rightPole = poles[1]
            else:
                leftPole = poles[1]
                rightPole = poles[0]
            params = getGateParameters(leftPole, rightPole, len(img_hsv[0]))
            if params == None:
                # Did not find gate even after discarding closest pole, probably invalid data
                continue
        params['color'] = COLOR_NAMES[color]
        gatesDetected.append(params)

    if len(gatesDetected) == 0:
        print('ERROR: No gates detected!')
        exit(100)
    gatesDetected = sorted(gatesDetected, key=lambda x: x.get('v'), reverse=False)
    nearestGate = gatesDetected[0]
    print "Color of nearest gate: ", nearestGate['color']
    print "c: ", nearestGate['c']
    print "g: ", nearestGate['g']
    print "alpha: ", nearestGate['alpha']
    print "beta: ", nearestGate['beta']
    print "v: ", nearestGate['v']
    print "d: ", nearestGate['d'] 
    print "Will robot pass after rotation?", nearestGate['willPass']
    return 

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('ERROR: No filename given! Please add the data filename as an argument while calling this script')
        exit(101)
    data = loadmat(sys.argv[1])
    #imsave('testimg.png',cv2.cvtColor(data['image_rgb'], cv2.COLOR_BGR2RGB))
    getNearestGateParameters(data)
