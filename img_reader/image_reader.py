import tensorflow as tf
import os
import glob


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
                image = tf.image.decode_jpeg(image)
                sess.run(image)
            except:
                print f + " is damaged"


def __read_file(input_queue, sampling_size):
    file_name = input_queue[0]
    label = input_queue[1]
    record = tf.read_file(file_name)
    image = __process_image(record, sampling_size)
    return image, label


def __process_image(record, sampling_size):
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


def input_pipeline(batch_size, sampling_size, min_after_dequeue, labels, indices, path='./training_set'):
    with tf.device('/cpu:0'):
        input_queue = __make_input_queue(labels, indices, path)
        images, lbs = __read_file(input_queue, sampling_size)
        capacity = min_after_dequeue + 3 * batch_size
        x_batch, y_batch = tf.train.shuffle_batch([images, lbs], batch_size, capacity, min_after_dequeue)

    return x_batch, y_batch
