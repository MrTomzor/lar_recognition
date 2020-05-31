import cv2
from scipy.io import loadmat
from scipy.misc import imsave

data = loadmat('2020-04-23-10-15-56.mat')
print(len(data['image_rgb'][0]))
#res = cv2.connectedComponentsWithStats(data['image_rgb'])
imsave('test_img.png', data['image_rgb'])
