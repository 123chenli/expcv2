import os
import tensorflow as tf
from PIL import Image

# 原始图像的存储位置
orig_picture = './orig_Images'

# 生成图片的存储位置
gen_picture = './gent_Images'

# 需要识别的类型
classes = {'husky', 'jiwawa', 'poodle', 'qiutian'}

# 样本总数
num_sample = 124

# 制作TFRecords数据
def create_record():
    writer = tf.python_io.TFRecordWriter('train.tfrecords')
    for index, name in enumerate(classes):
        class_path = os.getcwd() + '/orig_Images/' + name + "/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((64, 64))  # 设置需要转换的图片大小
            img_raw = img.tobytes()  # 将图片转化为原生的bytes
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                })
            )
            writer.write(example.SerializeToString())
    writer.close()


def read_and_decode(filename):
    # 创建文件队列，不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # 从文件队列中创建一个reader
    reader = tf.TFRecordReader()
    # reader 从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # 从serialized_example中得到特征
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [64, 64, 3])
    label = tf.cast(label, tf.int32)
    image, label = tf.train.shuffle_batch([img, label],
                                                    batch_size=32,
                                                    capacity=2000,
                                                    min_after_dequeue=1000)
    return image, label


if __name__ == '__main__':
    batch = read_and_decode('train.tfrecords')
    print(batch)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:  # 开始一个会话
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(num_sample):
            example, lab = sess.run(batch)
            img = Image.fromarray(example, 'RGB')
            img.save(os.getcwd() + '/gent_Images/' + str(i) + 'samples' + str(lab) + '.jpg')  # 存图片
        coord.request_stop()
        coord.join(threads)
        sess.close()
