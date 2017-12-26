import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 生成图片路径和标签的List
train_dir = 'D:\\file\gent_Images'

husky = []
label_husky = []
jiwawa = []
label_jiwawa = []
poodle = []
label_poodle = []
qiutian = []
label_qiutian = []


# 获取训练下的图片路径名，存放到对应的列表中，同时贴上标签，存放到label列表中
def get_files(file_dir, ratio):
    for file in os.listdir(file_dir + '/husky'):
        husky.append(file_dir + '/husky' + '/' + file)
        label_husky.append(0)
    for file in os.listdir(file_dir + '/jiwawa'):
        jiwawa.append(file_dir + '/jiwawa' + '/' + file)
        label_jiwawa.append(1)
    for file in os.listdir(file_dir + '/poodle'):
        poodle.append(file_dir + '/poodle' + '/' + file)
        label_poodle.append(2)
    for file in os.listdir(file_dir + '/qiutian'):
        qiutian.append(file_dir + '/qiutian' + '/' + file)
        label_qiutian.append(3)

    # 对生成的图片路径和标签list做打乱处理把cat和dog合起来组成一个list（img和lab)
    image_list = np.hstack((husky, jiwawa, poodle, qiutian))
    label_list = np.hstack((label_husky, label_jiwawa, label_poodle, label_qiutian))

    # 利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    # 将所有的img和lab转换成list
    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 0])

    # 将所有list分为两部分，一部分用来训练tra，一部分用来测试val
    # ratio是测试集的比例
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample*ratio))  # 测试样本数
    n_train = n_sample - n_val  # 训练样本数

    tra_images = all_image_list[0: n_train]
    tra_labels = all_label_list[0: n_train]
    # for i in tra_labels:
    #     print(i)
    tra_labels = [str(i) for i in tra_labels]
    val_images = all_image_list[n_train: -1]
    val_labels = all_label_list[n_train: -1]
    val_labels = [str(i) for i in val_labels]

    return tra_images, tra_labels, val_images, val_labels


# 生成batch
# 将上面生成的List传入get_batch(),转换类型，产生一个输入队列queue，因为img和lab
# 是分开的，所以使用tf.train.slice_input_producer()，然后用tf.read_file()从队列中读取图像
# image_W, image_H,:设置好固定的图像高度和宽度
# 设置batch_size:每个batch要放多少张图片
# capacity:一个队列最大多少
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # 转换类型
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])  # 从一个队列中读取图片

    # 将图像解码，对图像进行旋转、缩放、裁剪、归一化等操作，让计算除的模型更加健壮
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)

    # 生成batch
    # image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
    # label_batch: 4D tensor [batch_size], dtype = tf.int32
    image_batch, label_batch = tf.train.batch([Image, label],
                                              batch_size=batch_size,
                                              num_threads=32,
                                              capacity=capacity
                                              )
    # 重新排列label，行数为[batch_size]
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch
