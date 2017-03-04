# -*- coding=utf-8 -*-

import tensorflow as tf
import numpy as np
import csv

# Parameters
learning_rate = 0.005
display_step = 100

# Network Parameters
n_hidden_1 = 16
n_hidden_2 = 8
n_input = 20
n_classes = 2


def one_hot(a, length):
    b = np.zeros([length, 2])
    for i in range(length):
        if a[i] == 0:
            b[i][1] = 1
        else:
            b[i][0] = 1
    return b


# 打开刚才的序号、文件名、标签的csv文件，因为文件中间隔有1行，所以做一些处理
with open('./csv_file/seq.csv') as csvfile:
    reader = csv.reader(csvfile)
    your_list = list(reader)

data_ndarray = np.zeros((100, n_input + 1))
for one in your_list:
    if len(one):
        tmp_i = int(one[0])
        for i in range(n_input + 1):
            data_ndarray[tmp_i][i] = float(one[i + 2])


train_data = data_ndarray[:80, 1:]
train_label = data_ndarray[:80, 0:1]
train_label = one_hot(train_label, train_data.shape[0])

test_data = data_ndarray[80:, 1:]
test_label = data_ndarray[80:, 0:1]
test_label = one_hot(test_label, test_data.shape[0])

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder("float")


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer
# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
# Construct model
pred = multilayer_perceptron(x, weights, biases)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
saver = tf.train.Saver()
# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.device('/gpu:0'):
    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        for epoch in range(20000):
            _, c = sess.run([optimizer, cost], feed_dict={x: train_data, y: train_label})
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(c))
        print("Optimization Finished!")

        saver.save(sess, "./model_nn/model.ckpt", write_meta_graph=None)
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: test_data, y: test_label}))
        pred = sess.run(pred, feed_dict={x: test_data, y: test_label})
        print(pred)
        print(test_label)

