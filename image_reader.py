import tensorflow as tf
import os
import glob

image_type = 'png'


def detect_damaged_image_file(labels, path):
    image_list = []
    for label in labels:
        files = glob.glob(os.path.join(path, label, '*'))
        for f in files:
            image_list.append(f)
    with tf.Session() as sess:
        for f in image_list:
            try:
                image = tf.read_file(f)
                if image_type == 'jpg':
                    image = tf.image.decode_jpeg(image)
                elif image_type == 'png':
                    image = tf.image.decode_png(image)
                sess.run(image)
            except:
                print f + " is damaged"


def __read_file(input_queue, sampling_size):
    file_name = input_queue[0]
    label = input_queue[1]
    record = tf.read_file(file_name)
    if image_type is 'jpg':
        image = __process_image_jpg(record, sampling_size)
    elif image_type is 'png':
        try:
            image = __process_image_png(record, sampling_size)
        except:
            print file_name
    else:
        print "Image format is not supported : '%s'" %image_type
        return None, None
    return image, label


def __process_image_png(record, sampling_size):
    image = tf.image.decode_png(record, 3)
    image = tf.image.resize_images(image, (sampling_size, sampling_size))
    return image


def __process_image_jpg(record, sampling_size):
    image = tf.image.decode_jpeg(record, 3)
    image = tf.image.resize_images(image, (sampling_size, sampling_size))
    return image


def __read_labeled_image_list(labels, indices, path):
    image_list = []
    label_list = []
    for label in labels:
        files = glob.glob(os.path.join(path, label, '*'))
        for f in files:
            image_list.append(f)
            label_list.append(indices[label])
    return image_list, label_list


def __make_input_queue(labels, indices, path):
    image_list, label_list = __read_labeled_image_list(labels, indices, path)
    image_list = tf.convert_to_tensor(image_list, tf.string)
    label_list = tf.convert_to_tensor(label_list)
    input_queue = tf.train.slice_input_producer([image_list, label_list])
    return input_queue


def input_pipeline(batch_size, sampling_size, min_after_dequeue, labels, indices, is_shuffle=True, path='./training_set'):
    input_queue = __make_input_queue(labels, indices, path)
    images, lbs = __read_file(input_queue, sampling_size)
    capacity = min_after_dequeue + 3 * batch_size
    if is_shuffle:
        x_batch, y_batch = tf.train.shuffle_batch([images, lbs], batch_size, capacity, min_after_dequeue)
    else:
        x_batch, y_batch = tf.train.batch([images, lbs], batch_size, capacity=capacity, num_threads=1)

    return x_batch, y_batch
