import cv2
import numpy as np
import matplotlib.pyplot as plot


# 已知常量
# 相机对白纸的距离
# 暂时单位是厘米
KNOWN_DISTANCE = 32.0
KNOWN_WIDTH = 25.5


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # 初始化缩放比例，并获取图像尺寸
    dim = None
    (h, w) = image.shape[: 2]

    # 如果宽度和高度均为0，则返回原图
    if width is None and height is None:
        return image

    # 宽度为0
    if width is None:
        # 根据高度计算缩放比例
        r = height / float(h)
        dim = (int(w*r), height)

    # 高度为0
    else:
        # 根据宽度计算缩放比例
        r = width / float(w)
        dim = (int(w*r), width)

    # 缩放图像
    resized = cv2.resize(image, dim, interpolation=inter)

    # 返回缩放后的图像
    return resized


def find_marker(image):
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 将灰度图高斯模糊去除明显噪点，并进行边缘检测
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    edged = cv2.Canny(gray, 35, 125)
    binary, cnts, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # 计算出面积最大的轮廓
    c = max(cnts, key=cv2.contourArea)
    cv2.drawContours(image, c, -1, (0, 255, 0), 3)
    array = cv2.minAreaRect(c)
    return array


def distance_to_camera(knownWidth, focalLenght, perWidth):
    return (knownWidth * focalLenght) / perWidth


ImgToDetArray = ['./tupian/left_0.jpg', './tupian/right_0.jpg', './tupian/left_5.jpg']


def main():
    # 使用静态图片测试
    img = cv2.imread(ImgToDetArray[0])
    res = resize(img, width=764)  # 等比缩放
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # 根据一直距离的图片计算焦距
    marker = find_marker(res)
    print('焦距：', marker)
    # 计算出焦距
    focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
    print('focalLength: %f px' % focalLength )

    for im in ImgToDetArray:
        imgToDet = cv2.imread(im)
        imgToDetRes = resize(imgToDet, width=764)  # 等比缩放
        markerDet = find_marker(imgToDetRes)
        distance = distance_to_camera(KNOWN_WIDTH, focalLength, markerDet[1][0])
        print('%s distance: %f cm' % (im, distance))

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()