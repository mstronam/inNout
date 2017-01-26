import tensorflow as tf
import image_reader
import resNet
import os
import glob

class FoodRecognizer:
    def __init__(self,
                 learning_rate=0.05,
                 dropout_rate=0.5,
                 batch_size=64,
                 min_after_dequeue=1000,
                 iteration=10000,
                 sampling_size=227):
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.min_after_dequeue = min_after_dequeue
        self.iteration = iteration
        self.sampling_size = sampling_size

        self.labels = []
        labels = glob.glob(os.path.join('./training_set', '*'))
        for label in labels:
            self.labels.append(label.split('/')[-1])

        self.index = dict()
        for i in range(len(self.labels)):
            self.index[self.labels[i]] = [0 for _ in range(len(self.labels))]
            self.index[self.labels[i]][i] = 1

    def train(self):
        x_batch, y_batch = image_reader.input_pipeline(self.batch_size,
                                                       sampling_size=self.sampling_size,
                                                       min_after_dequeue=self.min_after_dequeue,
                                                       labels=self.labels,
                                                       indices=self.index)
        """model = resNet.model(x_batch,
                             is_training=True,
                             num_classes=len(self.labels))
        loss = resNet.loss(model, y_batch)
        predictions = tf.nn.softmax(model)

        in_top_1 = tf.nn.in_top_k(predictions, y_batch, 1)
        in_top_5 = tf.nn.in_top_k(predictions, y_batch, 5)

        correct_ratio_1 = tf.to_float(tf.reduce_sum(in_top_1)) / self.batch_size
        correct_ratio_5 = tf.to_float(tf.reduce_sum(in_top_5)) / self.batch_size"""

        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            for _ in xrange(self.iteration):
                print sess.run([x_batch, y_batch])
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    fr = FoodRecognizer()
    fr.train()
