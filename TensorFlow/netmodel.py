import tensorflow as tf

# 网络结构定义：
# 输入参数： images, image batch 、4D tensor、tf.float32、[batch_size, width, height, channels]
# 返回参数： logits, float, [batch_size, n_classes]
def inference(images, batch_size, n_classes):
    # 一个简单的卷积神经网络，卷积+池化层*2，全连接层*2，最后一个softmax层做分类
    # 卷积层1
    # 64个3&*3的卷积核（3通道），padding='SAME',表示padding后卷积的图与原图尺寸一直，激活函数relu（）
    with tf.variable_scope('conv1') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64],
                                                  stddev=1.0,
                                                  dtype=tf.float32),
                              name='weights',
                              dtype=tf.float32)
        biases = tf.Variable(tf.constant(value=0.1,
                                         dtype=tf.float32,
                                         shape=[64]),
                             name='biases',
                             dtype=tf.float32)
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias.add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # 池化层1
    # 3*3最大池化，步长strides为2，池化后执行lrn()操作，局部相应归一化，对训练有利
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                 padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radiu=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')

    # 卷积层2
    # 16个3*3的卷积核（16通道）,padding='SAME'，表示padding后卷积的图与原图尺寸一致，激活函数relu()
    with tf.variable_scope('conv2') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.Variable(tf.truncated_normal(shape=[dim, 128],
                                                    stddev=0.005,
                                                    dtype=tf.float32),
                                  name='weights',
                                  dtype=tf.float32)
        biases = tf.Variable(tf.constant(value=0.1,
                                             dtype=tf.float32,
                                             shape=[128]),
                                 name='biases',
                                 dtype=tf.floats32)
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # 池化层2
    # 3*3最大赤化，步长strides为2，池化后执行lrn()操作
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                               padding='SAME', name='pooling2')
    # 全连接层3
    # 128个神经元，将之前的pool层的输出reshape成一行，激活函数relu()
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.Variable(tf.truncated_normal(shape=[dim, 128],
                                                  stddev=0.005,
                                                  dtype=tf.float32),
                              name='weights',
                              dtype=tf.float32)
        biases = tf.Variable(tf.constant(value=0.1,
                                         dtype=tf.float32,
                                         shape=[128]),
                             name='biases',
                             dtype=tf.float32)
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # 全连层4
    # 128个神经网络，激活函数relu()
    with tf.variable_scope('local4') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[128, 128],
                                                  stddev=0.005,
                                                  dtype=tf.float32),
                              name='weights',
                              dtype=tf.flaot32)
        biases = tf.Variable(tf.constant(value=0.1,
                                         dtype=tf.float32,
                                         shape=[128]),
                             name='biases',
                             dtype=tf.float32)
        local4 = tf.nn.relu(tf.matmul(locla3, weights) + biases, name='local4')



