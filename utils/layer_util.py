import tensorflow as tf


def get_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def get_constant_variable(shape, value=0.1):
    initial = tf.constant(value, shape=shape)
    return tf.Variable(initial)


def conv2d(
        inputs,
        out_channels,
        kernel_size,
        scope,
        stride=[1, 1],
        padding='SAME',
        stddev=1e-1,
        activation_fn=tf.nn.relu):
    """
      Args:
        inputs: 4-D tensor variable BxHxWxC
        output_channels: int
        kernel_size: a list of 2 ints
        scope: string
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        stddev: float, stddev for truncated_normal init
        activation_fn: function

      Returns:
        Variable tensor
      """


    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        in_channels = inputs.shape[-1].value
        kernel_shape = [kernel_h, kernel_w, in_channels, out_channels]
        kernel = get_variable(kernel_shape, stddev)
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d(inputs, kernel, strides=[1, stride_h, stride_w, 1], padding=padding)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='SAME'):
    """ 2D max pooling.

      Args:
        inputs: 4-D tensor BxHxWxC
        kernel_size: a list of 2 ints
        stride: a list of 2 ints

      Returns:
        Variable tensor
      """

    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.max_pool(inputs,
                                 ksize=[1, kernel_h, kernel_w, 1],
                                 strides=[1, stride_h, stride_w, 1],
                                 padding=padding,
                                 )
        return outputs


def full_connection(
        inputs,
        num_outputs,
        scope,
        stddev=1e-1,
        activation_fn=tf.nn.relu):
    """
      Args:
        inputs: 2-D tensor BxN
        num_outputs: int

      Returns:
        Variable tensor of size B x num_outputs.
      """

    with tf.variable_scope(scope) as sc:
        num_inputs = inputs.shape[-1].value
        weights = get_variable(shape=[num_inputs, num_outputs], stddev=stddev)
        outputs = tf.matmul(inputs, weights)
        biases = get_constant_variable(shape=[num_outputs])
        outputs = tf.nn.bias_add(outputs, biases)
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.01)(weights))
        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def dropout(inputs, scope, keep_prob, is_training=False):
    with tf.variable_scope(scope) as sc:
        outputs = tf.nn.dropout(inputs, keep_prob=keep_prob)
        return outputs


