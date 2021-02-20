import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import numpy as np
import math
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

data.test.cls = np.argmax(data.test.labels, axis=1)
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10


x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

#  model one #
net1 = tf.layers.conv2d(inputs=x_image, name='layer1_conv1', padding='same',
                       filters=16, kernel_size=5, activation=tf.nn.relu)
net1 = tf.layers.max_pooling2d(inputs=net1, pool_size=2, strides=2)
net1 = tf.layers.conv2d(inputs=net1, name='layer1_conv2', padding='same',
                       filters=36, kernel_size=5, activation=tf.nn.relu)
net1 = tf.layers.max_pooling2d(inputs=net1, pool_size=2, strides=2)
net1 = tf.layers.flatten(net1)
net1 = tf.layers.dense(inputs=net1, name='layer1_fc',
                      units=32, activation=tf.nn.relu)
net1 = tf.layers.dense(inputs=net1, name='layer1_fc1',
                      units=64, activation=tf.nn.relu)
net1 = tf.layers.dense(inputs=net1, name='layer1_fc2',
                      units=128, activation=tf.nn.relu)
logits1 = tf.layers.dense(inputs=net1, name='layer1_fc_out',
                      units=num_classes, activation=None)
y1_pred = tf.nn.softmax(logits=logits1)
y1_pred_cls = tf.argmax(y1_pred, dimension=1)

cross_entropy1 = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits1)
loss1 = tf.reduce_mean(cross_entropy1)
opt1 = tf.train.AdamOptimizer(learning_rate=1e-4)
optimizer1 = opt1.minimize(loss1)
correct_prediction1 = tf.equal(y1_pred_cls, y_true_cls)
accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))

# model two #
net2 = tf.layers.conv2d(inputs=x_image, name='layer2_conv1', padding='same',
                       filters=16, kernel_size=5, activation=tf.nn.relu)
net2 = tf.layers.max_pooling2d(inputs=net2, pool_size=2, strides=2)
net2 = tf.layers.conv2d(inputs=net2, name='layer2_conv2', padding='same',
                       filters=36, kernel_size=5, activation=tf.nn.relu)
net2 = tf.layers.max_pooling2d(inputs=net2, pool_size=2, strides=2)
net2 = tf.layers.flatten(net2)
net2 = tf.layers.dense(inputs=net2, name='layer2_fc',
                      units=32, activation=tf.nn.relu)
logits2 = tf.layers.dense(inputs=net2, name='layer2_fc_out',
                      units=num_classes, activation=None)
y2_pred = tf.nn.softmax(logits=logits2)
y2_pred_cls = tf.argmax(y2_pred, dimension=1)

cross_entropy2 = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits2)
loss2 = tf.reduce_mean(cross_entropy2)
opt2 = tf.train.AdamOptimizer(learning_rate=1e-4)
optimizer2 = opt1.minimize(loss2)
correct_prediction2 = tf.equal(y2_pred_cls, y_true_cls)
accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))

total_iterations = 0
train_batch_size = 256
loss1_list=[]
acc1_list=[]
loss2_list=[]
acc2_list=[]
def optimize(num_iterations):
    global total_iterations
    for i in range(total_iterations,
                   total_iterations + num_iterations):
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        session.run(optimizer1, feed_dict=feed_dict_train)
        if i % 1 == 0:
            los1, acc1 = session.run([loss1, accuracy1], feed_dict=feed_dict_train)
            loss1_list.append(los1)
            acc1_list.append(acc1)
            msg = "Iteration1: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}"
            print(msg.format(i + 1, los1, acc1))
        session.run(optimizer2, feed_dict=feed_dict_train)
        if i % 1 == 0:
            los2, acc2 = session.run([loss2, accuracy2], feed_dict=feed_dict_train)
            loss2_list.append(los2)
            acc2_list.append(acc2)
            msg = "Iteration2: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}"
            print(msg.format(i + 1, los2, acc2))
    total_iterations += num_iterations


session = tf.Session()
session.run(tf.global_variables_initializer())

optimize(num_iterations=1000)
plt.plot(loss1_list,'b-',loss2_list,'r-')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Iterations')
plt.savefig('loss.jpg')
plt.show()
plt.plot(acc1_list,'b-',acc2_list,'r-')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.savefig('acc.jpg')
plt.show()






