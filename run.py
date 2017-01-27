from resNet_50 import *
import tensorflow as tf
import image_reader
import os
import glob
import time
import sys

MOMENTUM = 0.9

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('learning_rate', 0.01, "learning rate.")
tf.app.flags.DEFINE_boolean('resume', False, "learning rate.")
tf.app.flags.DEFINE_integer('iteration', 100000, "max steps")
tf.app.flags.DEFINE_integer('batch_size', 4, "batch size")
tf.app.flags.DEFINE_integer('min_after_dequeue', 100, "amount of images stored in queue")
tf.app.flags.DEFINE_integer('sampling_size', 512, "sampling_size")
tf.app.flags.DEFINE_string('training_set_path', './training_set', "path of training data set")
tf.app.flags.DEFINE_string('test_set_path', './test_set', "path of test data set")
tf.app.flags.DEFINE_string('data_path', './data', "path of files needed to train")
tf.app.flags.DEFINE_string('log_path', './log', "path of files needed to train")


def top_k_error(predictions, labels, k):
    batch_size = float(FLAGS.batch_size)
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k))
    num_correct = tf.reduce_sum(in_top1)
    return (batch_size - num_correct) / batch_size

labels = []
ls = glob.glob(os.path.join(FLAGS.training_set_path, '*'))
for l in ls:
    labels.append(l.split('/')[-1])

indices = dict()
for i in range(len(labels)):
    indices[labels[i]] = i

def train():

    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    val_step = tf.get_variable('val_step', [],
                               initializer=tf.constant_initializer(0),
                               trainable=False)

    x_batch, y_batch = image_reader.input_pipeline(FLAGS.batch_size,
                                                   sampling_size=FLAGS.sampling_size,
                                                   min_after_dequeue=FLAGS.min_after_dequeue,
                                                   labels=labels,
                                                   indices=indices)

    model_ = model(x_batch,
                   is_training=True,
                   num_classes=len(labels))
    loss_ = loss(model_, y_batch)
    predictions = tf.nn.softmax(model_)

    correct_in_1 = tf.nn.in_top_k(predictions, y_batch, k=1)
    correct_in_5 = tf.nn.in_top_k(predictions, y_batch, k=5)

    accuracy_in_1 = tf.reduce_mean(tf.cast(correct_in_1, tf.float32))
    accuracy_in_5 = tf.reduce_mean(tf.cast(correct_in_5, tf.float32))

    top1_error = top_k_error(predictions, y_batch, 1)

    # loss_avg
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([loss_]))
    tf.summary.scalar('loss_avg', ema.average(loss_))

    # validation stats
    ema = tf.train.ExponentialMovingAverage(0.9, val_step)
    val_op = tf.group(val_step.assign_add(1), ema.apply([top1_error]))
    top1_error_avg = ema.average(top1_error)
    tf.summary.scalar('val_top1_error_avg', top1_error_avg)

    tf.summary.scalar('learning_rate', FLAGS.learning_rate)

    opt = tf.train.MomentumOptimizer(FLAGS.learning_rate, MOMENTUM)
    grads = opt.compute_gradients(loss_)
    for grad, var in grads:
        tf.summary.histogram(var.op.name + '/gradients', grad)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

    saver = tf.train.Saver(tf.global_variables())

    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        summary_writer = tf.summary.FileWriter(FLAGS.log_path, sess.graph)

        if FLAGS.resume:
            latest = tf.train.latest_checkpoint(FLAGS.data_path)
            if not latest:
                print "No checkpoint to continue from in", FLAGS.data_path
                sys.exit(1)
            print "resume", latest
            saver.restore(sess, latest)

        for _ in xrange(FLAGS.iteration):
            start_time = time.time()

            step = sess.run(global_step)
            i = [train_op, loss_, accuracy_in_1, accuracy_in_5]

            write_summary = step % 100 and step > 1
            if write_summary:
                i.append(summary_op)

            o = sess.run(i)

            loss_value = o[1]
            acc_1 = o[2]
            acc_5 = o[3]

            duration = time.time() - start_time

            if step % 5 == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                format_str = ('step %d, loss = %.2f top 1 accuracy = %.4f, top 5 accuracy = %.4f (%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (step, loss_value, acc_1, acc_5, examples_per_sec, duration))

            if write_summary:
                summary_str = o[4]
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step > 1 and step % 100 == 0:
                checkpoint_path = os.path.join(FLAGS.data_path, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=global_step)

            # Run validation periodically
            if step > 1 and step % 100 == 0:
                _, top1_error_value = sess.run([val_op, top1_error])
                print('Validation top1 error %.2f' % top1_error_value)

            coord.request_stop()
            coord.join(threads)


def test():
    total_amount = 0

    test_labels = []
    ls = glob.glob(os.path.join(FLAGS.test_set_path, '*'))
    for l in ls:
        test_labels.append(l.split('/')[-1])
        fs = glob.glob(os.path.join(l, '*'))
        total_amount += len(fs)

    x_batch, y_batch = image_reader.input_pipeline(batch_size=1,
                                                   sampling_size=FLAGS.sampling_size,
                                                   min_after_dequeue=total_amount,
                                                   labels=test_labels,
                                                   indices=indices,
                                                   is_shuffle=False,
                                                   path='./test_set')

    model_ = model(x_batch,
                   is_training=False,
                   num_classes=len(labels))

    predictions = tf.nn.softmax(model_)

    correct_in_1 = tf.nn.in_top_k(predictions, y_batch, k=1)
    correct_in_5 = tf.nn.in_top_k(predictions, y_batch, k=5)

    accuracy_in_1 = tf.reduce_mean(tf.cast(correct_in_1, tf.float32))
    accuracy_in_5 = tf.reduce_mean(tf.cast(correct_in_5, tf.float32))

    acc_1 = 0
    acc_5 = 0
    acc_cat = [[0, 0] for _ in range(len(labels))]

    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        latest = tf.train.latest_checkpoint(FLAGS.data_path)
        if not latest:
            print "No checkpoint to continue from in", FLAGS.data_path
            sys.exit(1)
        print "resume", latest
        saver.restore(sess, latest)

        for step in xrange(total_amount):
            while True:
                try:
                    o = sess.run([accuracy_in_1, accuracy_in_5, correct_in_1, correct_in_5, y_batch])
                    break
                except:
                    time.sleep(5)
                    continue
            acc_1 += o[0]
            acc_5 += o[1]
            acc_cat[int(o[4])][0] += o[2]
            acc_cat[int(o[4])][1] += o[3]

            if step % 10 == 0:
                progress = (float(step) / total_amount) * 100
                print"Testing Progress : %.2f" % (progress)

        coord.request_stop()
        coord.join(threads)

    acc_1_mean = acc_1 / float(total_amount)
    acc_5_mean = acc_5 / float(total_amount)
    for k in range(len(acc_cat)):
        item_mean = [0, 0]
        item_mean[0] = acc_cat[k][0] / float(total_amount)
        item_mean[1] = acc_cat[k][1] / float(total_amount)
        acc_cat[k] = item_mean

    __print_test_result(acc_1_mean, acc_5_mean)
    __write_test_result(acc_1_mean, acc_5_mean, acc_cat)


def __print_test_result(acc_1, acc_5):
    print("**************************************************")
    print("*          T E S T          R E S U L T          *")
    print("**************************************************")
    print("**************************************************")
    print("*   ACCURACY (TOP IN 1)    :    %.4f           *" %(acc_1))
    print("*   ACCURACY (TOP IN 5)    :    %.4f           *" %(acc_5))
    print("**************************************************")


def __write_test_result(acc_1, acc_5, acc_cat):
    current_time = time.strftime('%y%m%d%H%M', time.gmtime())
    result_file = open('result_%s' %current_time , 'w+')
    result_file.write("**************************************************\n")
    result_file.write("*          T E S T          R E S U L T          *\n")
    result_file.write("**************************************************\n")
    result_file.write("**************************************************\n")
    result_file.write("*   TOTAL ACCURACY (TOP IN 1)    :    %.4f     *\n" %(acc_1))
    result_file.write("*   TOTAL ACCURACY (TOP IN 5)    :    %.4f     *\n" %(acc_5))
    result_file.write("**************************************************\n\n")
    for k in range(len(acc_cat)):
        food_name = labels[k]
        result_file.write("%s ACCURACY (TOP IN 1)    :\t\t    %.4f\n" % (food_name, acc_cat[k][0]))
        result_file.write("%s ACCURACY (TOP IN 5)    :\t\t    %.4f\n\n" % (food_name, acc_cat[k][1]))

    result_file.close()


if __name__ == '__main__':
    test()
