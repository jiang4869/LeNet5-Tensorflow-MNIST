import tensorflow as tf
from moudle import lenet
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data/MNIST_data", one_hot=True)

BATCH_SIZE = 100

N_BATCH = mnist.train.num_examples // BATCH_SIZE


def train():
    with tf.variable_scope('inputs'):
        x = tf.placeholder(tf.float32, [None, 784])
        y = tf.placeholder(tf.float32, [None, 10])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    outputs, prediction = lenet.get_model(x_image)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    init = tf.global_variables_initializer()

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs/', sess.graph)
        sess.run(init)
        for epoch in range(20):
            for batch in range(N_BATCH):
                batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
                sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
            pre = sess.run(prediction, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print('Iter' + str(epoch) + ",Testing Accuracy " + str(acc))
        saver.save(sess, 'logs/train.ckpt')


if __name__ == '__main__':
    train()
