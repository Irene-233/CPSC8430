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

trainable_var_list = tf.trainable_variables()
def get_weights_variable(layer_name):
    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('kernel')
    return variable
weights1_conv1 = get_weights_variable(layer_name='layer1_conv1')
weights1_conv2 = get_weights_variable(layer_name='layer1_conv2')

weights1_fc1 = get_weights_variable(layer_name='layer1_fc')
weights1_fc_out = get_weights_variable(layer_name='layer1_fc_out')




# model two #
net2 = tf.layers.conv2d(inputs=x_image, name='layer2_conv1', padding='same',
                       filters=16, kernel_size=5, activation=tf.nn.relu)
net2 = tf.layers.max_pooling2d(inputs=net2, pool_size=2, strides=2)
net2 = tf.layers.conv2d(inputs=net2, name='layer2_conv2', padding='same',
                       filters=36, kernel_size=5, activation=tf.nn.relu)
net2 = tf.layers.max_pooling2d(inputs=net2, pool_size=2, strides=2)
net2 = tf.layers.flatten(net2)
net2 = tf.layers.dense(inputs=net2, name='layer2_fc',
                      units=128, activation=tf.nn.relu)
logits2 = tf.layers.dense(inputs=net2, name='layer2_fc_out',
                      units=num_classes, activation=None)
y2_pred = tf.nn.softmax(logits=logits2)
y2_pred_cls = tf.argmax(y2_pred, dimension=1)

cross_entropy2 = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits2)
loss2 = tf.reduce_mean(cross_entropy2)
opt2 = tf.train.AdamOptimizer(learning_rate=1e-2)
optimizer2 = opt2.minimize(loss2)
correct_prediction2 = tf.equal(y2_pred_cls, y_true_cls)
accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
trainable_var_list = tf.trainable_variables()
weights2_conv1 = get_weights_variable(layer_name='layer2_conv1')
weights2_conv2 = get_weights_variable(layer_name='layer2_conv2')

weights2_fc1 = get_weights_variable(layer_name='layer2_fc')
weights2_fc_out = get_weights_variable(layer_name='layer2_fc_out')

total_iterations = 0
train_batch_size = 256
loss1_list=[]
acc1_list=[]
weights1_1=0
weights1_2=0
weights1_3=0
weights1_4=0

weights2_1=0
weights2_2=0
weights2_3=0
weights2_4=0

def optimize(num_iterations):
    global total_iterations
    global weights1_1
    global weights1_2
    global weights1_3
    global weights1_4
    global weights2_1
    global weights2_2
    global weights2_3
    global weights2_4
    for i in range(total_iterations,
                   total_iterations + num_iterations):
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        session.run(optimizer1, feed_dict=feed_dict_train)
        if i % 1 == 0:
            los1, acc1 = session.run([loss1, accuracy1], feed_dict=feed_dict_train)
        if i == num_iterations-1: 
            weights1_1, weights1_2,weights1_3,weights1_4 = session.run([weights1_conv1, weights1_conv2,weights1_fc1,weights1_fc_out], feed_dict=feed_dict_train)  
        session.run(optimizer2, feed_dict=feed_dict_train)
        if i % 1 == 0:
            los2, acc2 = session.run([loss2, accuracy2], feed_dict=feed_dict_train)
        if i == num_iterations-1: 
            weights2_1, weights2_2,weights2_3,weights2_4 = session.run([weights2_conv1, weights2_conv2,weights2_fc1,weights2_fc_out], feed_dict=feed_dict_train)     
            
    total_iterations += num_iterations
    


session = tf.Session()
session.run(tf.global_variables_initializer())

fd={x: data.train.images,y_true: data.train.labels}
optimize(num_iterations=5)

def getalpha(alpha):
  weights_conv1=weights1_1*(1-alpha)+weights2_1*alpha
  weights_conv2=weights1_2*(1-alpha)+weights2_2*alpha
  weights_fc=weights1_3*(1-alpha)+weights2_3*alpha
  weights_fcout=weights1_4*(1-alpha)+weights2_4*alpha
  weights1_conv1.assign(tf.Variable(weights_conv1))
  weights2_conv2.assign(tf.Variable(weights_conv2))
  weights2_fc1.assign(tf.Variable(weights_fc))
  weights2_fc_out.assign(tf.Variable(weights_fcout))
  session.run(optimizer1, feed_dict=fd)
  los1, acc1 = session.run([loss1, accuracy1],feed_dict=fd)
  loss1_list.append(los1)
  acc1_list.append(acc1)

getalpha(-1)
getalpha(-0.5)
getalpha(0)
getalpha(0.5)
getalpha(1)
getalpha(1.5)
getalpha(2)
print(loss1_list)
print(acc1_list)
num=[-1,-0.5,0,0.5,1,1.5,2]
x = np.arange(0, len(num))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(loss1_list,'b--', label = 'loss')
ax2 = ax.twinx()
ax2.plot(acc1_list,'r--', label = 'acc')
ax.legend(loc=0)
ax.grid()
ax.set_xlabel("alpha")
ax.set_ylabel("loss")
ax2.set_ylabel("acc")
ax2.set_ylim(0.2,0.5)
ax.set_ylim(2.1, 2.5)
ax2.legend(loc=0)
plt.xticks(x, num)

plt.savefig('alpha.jpg')
plt.show()



