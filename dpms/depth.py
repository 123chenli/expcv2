import numpy as np
import cv2
from matplotlib import pyplot as plt


def createMedianMask(disparityMap, validDepthMask, rect=None):
    if rect is not None:
        x, y, w, h = rect
        disparityMap = disparityMap[y: y+h, x: x+w]
        validDepthMask = validDepthMask[y: y+h, x: x+w]
    median = np.median(disparityMap)
    return np.where((validDepthMask==0) |
                       (abs(disparityMap - median) < 12),
                       1.0, 0.0)


def to_uint8(data):
    latch = np.zeros_like(data)
    latch[:] = 255
    zeros = np.zeros_like(data)
    d = np.maximum(zeros, data)
    d = np.minimum(latch, d)
    return np.asarray(d, dtype='uint8')


def drawlines(img1, img2, lines, pts1, pts2):
    r, c, ch = img1.shape
    print(cv2.imread('./images/stacked1.png'))
    clr1 = cv2.pyrDown(cv2.imread('../images/stacked1.png'))
    clr2 = cv2.pyrDown(cv2.imread('../images/stacked2.png'))
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        clr1 = cv2.line(clr1, (x0, y0), (x1, y1), color, 1)
        clr1 = cv2.circle(clr1, tuple(pt1), 5, color, -1)
        clr2 = cv2.circle(clr2, tuple(pt2), 5, color, -1)
        return clr1, clr2

img1= to_uint8(cv2.pyrDown(cv2.imread('../images/stacked1.png')))
img2 = to_uint8(cv2.pyrDown(cv2.imread('../images/stacked2.png')))
sift = cv2.SIFT()


kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, tree=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = []
pts1 = []
pts2 = []

for i, (m, n) in enumerate(matches):
    if m.distanch < 0.8 * n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)


pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

pts1 = pts1[mask.reval() == 1]
pts2 = pts2[mask.raval() == 1]

lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 2, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

plt.subplot(121)
plt.imshow(img5)
plt.subplot(122)
plt.imshow(img3)
plt.show()