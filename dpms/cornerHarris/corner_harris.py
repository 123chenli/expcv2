import cv2
import numpy as np

img = cv2.imread('chess_board.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 23, 0.04)
img[dst > 0.01 * dst.max()] = [0, 0, 255]  # 将检测到的角点标记为红色
while True:
    cv2.imshow('corners', img)
    if cv2.waitKey() & 0xff == ord('q'):
        break
cv2.destroyAllWindows()
