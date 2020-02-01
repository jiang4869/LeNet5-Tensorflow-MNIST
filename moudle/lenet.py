import tensorflow as tf
from utils import layer_util


def LeNet(inputs, keep_prob=None):
    with tf.variable_scope('Conv1'):
        conv1 = layer_util.conv2d(inputs, 6, [5, 5], 'conv1', padding='VALID')
    with tf.variable_scope('S2'):
        s2 = layer_util.max_pool2d(conv1, [2, 2], 'S2')
    with tf.variable_scope('Conv3'):
        conv3 = layer_util.conv2d(s2, 16, [5, 5], 'conv3', padding='VALID')
    with tf.variable_scope('S4'):
        s4 = layer_util.max_pool2d(conv3, [2, 2], 's4')
    with tf.variable_scope('Conv5'):
        conv5 = layer_util.conv2d(s4, 120, [5, 5], 'conv5')
    flattened_shape = conv5.shape[1].value * conv5.shape[2].value * conv5.shape[3].value
    conv5 = tf.reshape(conv5, [-1, flattened_shape])
    with tf.variable_scope('F6'):
        f6 = layer_util.full_connection(conv5, 84, 'f6')
    with tf.variable_scope('output'):
        outputs = layer_util.full_connection(f6, 10, 'outputs', activation_fn=None)
    prediction = tf.nn.softmax(outputs)
    return outputs,prediction


def get_model(inputs, keep_prob=None):
    return LeNet(inputs, keep_prob)
