import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops


CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'


def model(x,
          is_training,
          num_classes=1000,
          num_blocks=[3, 4, 6, 3],
          use_bias=False,
          bottle_neck=True):

    c = {'bottle_neck': bottle_neck,
         'is_training': tf.convert_to_tensor(is_training, dtype=tf.bool, name='is_training'),
         'ksize': 3,
         'stride': 1,
         'use_bias': use_bias,
         'fc_units_out': num_classes,
         'num_blocks': num_blocks,
         'stack_stride': 2}

    with tf.variable_scope('scale1'):
        c['conv_filters_out'] = 64
        c['ksize'] = 7
        c['stride'] = 2
        x = __conv(x, c)
        x = __bn(x, c)
        x = __activation(x)

    with tf.variable_scope('scale2'):
        x = __max_pool(x, ksize=3, stride=2)
        c['num_blocks'] = num_blocks[0]
        c['stack_stride'] = 1
        c['block_filters_internal'] = 64
        x = stack(x, c)

    with tf.variable_scope('scale3'):
        c['num_blocks'] = num_blocks[1]
        c['block_filters_internal'] = 128
        x = stack(x, c)

    with tf.variable_scope('scale4'):
        c['num_blocks'] = num_blocks[2]
        c['block_filters_internal'] = 256
        x = stack(x, c)

    with tf.variable_scope('scale5'):
        c['num_blocks'] = num_blocks[3]
        c['block_filters_internal'] = 512
        x = stack(x, c)

    x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")

    if num_classes is not None:
        with tf.variable_scope('fc'):
            x = __fc(x, c)

    return x


def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
    tf.summary.scalar('loss', loss_)

    return loss_


def stack(x, c):
    for n in range(c['num_blocks']):
        s = c['stack_stride'] if n == 0 else 1
        c['block_stride'] = s
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(x, c)
    return x


def block(x, c):
    filters_in = x.get_shape()[-1]

    m = 4 if c['bottle_neck'] else 1
    filters_out = m * c['block_filters_internal']

    shortcut = x

    c['conv_filters_out'] = c['block_filters_internal']

    if c['bottle_neck']:
        with tf.variable_scope('a'):
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            x = __conv(x, c)
            x = __bn(x, c)
            x = __activation(x)

        with tf.variable_scope('b'):
            x = __conv(x, c)
            x = __bn(x, c)
            x = __activation(x)

        with tf.variable_scope('c'):
            c['conv_filters_out'] = filters_out
            c['ksize'] = 1
            assert c['stride'] == 1
            x = __conv(x, c)
            x = __bn(x, c)

    else :
        with tf.variable_scope('A'):
            c['stride'] = c['block_stride']
            assert c['ksize'] == 3
            x = __conv(x, c)
            x = __bn(x, c)
            x = __activation(x)

        with tf.variable_scope('B'):
            c['conv_filter_out'] = filters_out
            assert c['ksize'] == 3
            assert c['stride'] == 1
            x = __conv(x, c)
            x = __bn(x, c)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or c['block_stride'] != 1:
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            c['conv_filters_out'] = filters_out
            shortcut = __conv(shortcut, c)
            shortcut = __bn(shortcut, c)

    return __activation(x + shortcut)


def __get_variable(name,
                   shape,
                   initializer,
                   weight_decay=0.0,
                   dtype='float',
                   trainable=True):
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)


def __conv(x, c):
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']
    filters_in = x.get_shape()[-1]

    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = __get_variable('weights',
                             shape=shape,
                             initializer=initializer,
                             weight_decay=CONV_WEIGHT_DECAY)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


def __max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')


def __activation(x):
    return tf.nn.relu(x)


def __fc(x, c):
    num_units_out = c['fc_units_out']
    num_units_in = x.get_shape()[-1]
    initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV)
    weights = __get_variable('weights',
                             shape=[num_units_in, num_units_out],
                             initializer=initializer,
                             weight_decay=FC_WEIGHT_DECAY)
    biases = __get_variable('biases',
                            shape=[num_units_out],
                            initializer=tf.zeros_initializer)
    return tf.nn.xw_plus_b(x, weights, biases)


def __bn(x, c):
    x_shape = x.get_shape()
    para_shape = x_shape[-1:]

    if c['use_bias']:
        bias = __get_variable('bias',
                              shape=shape,
                              initializer=tf.zeros_initializer)
        return x + bias

    axis = list(range(len(x_shape) - 1))

    beta = __get_variable('beta',
                          shape=para_shape,
                          initializer=tf.zeros_initializer)
    gamma = __get_variable('gamma',
                           shape=para_shape,
                           initializer=tf.ones_initializer())
    moving_mean = __get_variable('moving_mean',
                                 shape=para_shape,
                                 initializer=tf.zeros_initializer,
                                 trainable=False)
    moving_variance = __get_variable('moving_variance',
                                     shape=para_shape,
                                     initializer=tf.ones_initializer(),
                                     trainable=False)

    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)

    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(c['is_training'],
                                           lambda: (mean, variance),
                                           lambda: (moving_mean, moving_variance))

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
