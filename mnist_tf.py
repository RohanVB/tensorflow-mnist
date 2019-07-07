from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import random as ran
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf


def train_size(num):
    """
    load the training set
    """
    x_train = mnist.train.images[:num, :]
    y_train = mnist.train.labels[:num, :]
    return x_train, y_train


def test_size(num):
    """
    load the test set
    """
    x_test = mnist.test.images[:num, :]
    y_test = mnist.test.labels[:num, :]
    return x_test, y_test


def display_digit(num):
    """
    show the image of the digit in grayscale, example number,
    label number, 28x28 pixels
    """
    print(y_train[num])
    label = y_train[num].argmax(axis=0)  # returns the label of the image
    image = x_train[num].reshape([28, 28])  # return the actual image
    plt.title('Example number : %d  Prediction: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()


# display_digit(2)


"""
placeholder is a variable to feed data into. in order to feed data into this variable we need to match its shape and type.
it is not initialised and it contains no data. It exists to serve as the target of feeds.
'None' assigned to the placeholder means that we can feed any amount of values that we want to it. (55000)
"""
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

"""
Initialize weights and bias to 0.
"""

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


"""
matmul is matrix multiplication of x and W.
"""
y = tf.nn.softmax(tf.matmul(x,W) + b)

"""
If the log function for each value is close to zero, it will make the value a large negative number
(i.e., -np.log(0.01) = 4.6), and if it is close to 1, it will make the value a small negative number
(i.e., -np.log(0.99) = 0.1).
"""
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


x_train, y_train = train_size(5500)
x_test, y_test = test_size(10000)

LEARNING_RATE = 0.01  # Learning rate set to small value
TRAIN_STEPS = 4000

"""
tensorflow creates a directed acyclic graph which we feed with data and run in a session.
"""
sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

# in the optimizer we enter learning rate and minimize cross entropy
training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

#  boolean
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

#  cast converts tensor to desired type (float32)
#  reduce_mean calculates mean in the same way as numpy.mean()
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(TRAIN_STEPS+1):
    sess.run(training, feed_dict={x: x_train, y_: y_train})
    if i % 100 == 0:  # print the training step, accuracy and cross entropy after every 100 training steps
        print('Training Step:' + str(i) + '  Accuracy =  ' + str(sess.run(accuracy, feed_dict={x: x_test, y_: y_test})) + '  Loss = ' + str(sess.run(cross_entropy, {x: x_train, y_: y_train})))

#  outputs a 1x10 matrix with each column as a probability
answer = sess.run(y, feed_dict={x: x_train})

#  gives the highest value
answer.argmax()


def display_compare(num):
    # Load a training example
    x_train = mnist.train.images[num, :].reshape(1, 784)
    y_train = mnist.train.labels[num, :]
    # load label as integer
    label = y_train.argmax()
    # load prediction as integer
    prediction = sess.run(y, feed_dict={x: x_train}).argmax()
    plt.title('Prediction: %d Label: %d' % (prediction, label))
    plt.imshow(x_train.reshape([28, 28]), cmap=plt.get_cmap('gray_r'))
    plt.show()


#  take a random image from the dataset and check if the prediction matches the label
display_compare(ran.randint(0, 55000))
