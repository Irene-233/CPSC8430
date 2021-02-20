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
                      units=64, activation=tf.nn.relu)
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
                      units=128, activation=tf.nn.relu)
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

# model three #

net3 = tf.layers.conv2d(inputs=x_image, name='layer3_conv1', padding='same',
                       filters=16, kernel_size=5, activation=tf.nn.relu)
net3 = tf.layers.max_pooling2d(inputs=net3, pool_size=2, strides=2)
net3 = tf.layers.conv2d(inputs=net3, name='layer3_conv2', padding='same',
                       filters=36, kernel_size=5, activation=tf.nn.relu)
net3 = tf.layers.max_pooling2d(inputs=net3, pool_size=2, strides=2)

net3 = tf.layers.flatten(net3)
net3 = tf.layers.dense(inputs=net3, name='layer3_fc',
                      units=256, activation=tf.nn.relu)
logits3 = tf.layers.dense(inputs=net3, name='layer3_fc_out',
                      units=num_classes, activation=None)
y3_pred = tf.nn.softmax(logits=logits3)
y3_pred_cls = tf.argmax(y3_pred, dimension=1)

cross_entropy3 = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits3)
loss3 = tf.reduce_mean(cross_entropy3)
opt3 = tf.train.AdamOptimizer(learning_rate=1e-4)
optimizer3 = opt1.minimize(loss3)
correct_prediction3 = tf.equal(y3_pred_cls, y_true_cls)
accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3, tf.float32))

# model four #

net4 = tf.layers.conv2d(inputs=x_image, name='layer4_conv1', padding='same',
                       filters=16, kernel_size=5, activation=tf.nn.relu)
net4 = tf.layers.max_pooling2d(inputs=net4, pool_size=2, strides=2)
net4 = tf.layers.conv2d(inputs=net4, name='layer4_conv2', padding='same',
                       filters=36, kernel_size=5, activation=tf.nn.relu)
net4 = tf.layers.max_pooling2d(inputs=net4, pool_size=2, strides=2)

net4 = tf.layers.flatten(net4)
net4 = tf.layers.dense(inputs=net4, name='layer4_fc',
                      units=512, activation=tf.nn.relu)                      
logits4 = tf.layers.dense(inputs=net4, name='layer4_fc_out',
                      units=num_classes, activation=None)
y4_pred = tf.nn.softmax(logits=logits4)
y4_pred_cls = tf.argmax(y4_pred, dimension=1)

cross_entropy4 = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits4)
loss4 = tf.reduce_mean(cross_entropy4)
opt4 = tf.train.AdamOptimizer(learning_rate=1e-4)
optimizer4 = opt1.minimize(loss4)
correct_prediction4 = tf.equal(y4_pred_cls, y_true_cls)
accuracy4 = tf.reduce_mean(tf.cast(correct_prediction4, tf.float32))


# model five#

net5 = tf.layers.conv2d(inputs=x_image, name='layer5_conv1', padding='same',
                       filters=16, kernel_size=5, activation=tf.nn.relu)
net5 = tf.layers.max_pooling2d(inputs=net5, pool_size=2, strides=2)
net5 = tf.layers.conv2d(inputs=net5, name='layer5_conv2', padding='same',
                       filters=36, kernel_size=5, activation=tf.nn.relu)
net5 = tf.layers.max_pooling2d(inputs=net5, pool_size=2, strides=2)

net5 = tf.layers.flatten(net5)
net5 = tf.layers.dense(inputs=net5, name='layer5_fc',
                      units=1024, activation=tf.nn.relu)
                                            
logits5 = tf.layers.dense(inputs=net5, name='layer5_fc_out',
                      units=num_classes, activation=None)
y5_pred = tf.nn.softmax(logits=logits5)
y5_pred_cls = tf.argmax(y5_pred, dimension=1)

cross_entropy5 = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits5)
loss5 = tf.reduce_mean(cross_entropy5)
opt5 = tf.train.AdamOptimizer(learning_rate=1e-4)
optimizer5 = opt1.minimize(loss5)
correct_prediction5 = tf.equal(y5_pred_cls, y_true_cls)
accuracy5 = tf.reduce_mean(tf.cast(correct_prediction5, tf.float32))

trainable_var_list = tf.trainable_variables()
def get_weights_variable(layer_name):
    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('kernel')
    return variable

weights1= get_weights_variable(layer_name='layer1_fc_out')
weights2= get_weights_variable(layer_name='layer2_fc_out')
weights3= get_weights_variable(layer_name='layer3_fc_out')
weights4= get_weights_variable(layer_name='layer4_fc_out')
weights5= get_weights_variable(layer_name='layer5_fc_out')


session = tf.Session()
session.run(tf.global_variables_initializer())
grad1 = tf.gradients(loss1, weights1)[0]
grad2 = tf.gradients(loss2, weights2)[0]
grad3 = tf.gradients(loss3, weights3)[0]
grad4 = tf.gradients(loss4, weights4)[0]
grad5 = tf.gradients(loss5, weights5)[0]


total_iterations = 0
train_batch_size = 128
loss1_list=[]
acc1_list=[]
sens_list=[]
def optimize(num_iterations):
    global total_iterations
    for i in range(total_iterations,
                   total_iterations + num_iterations):
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        session.run(optimizer1, feed_dict=feed_dict_train)
        los1, acc1 = session.run([loss1, accuracy1], feed_dict=feed_dict_train)
        grads_vals1 = session.run([grad1], feed_dict=feed_dict_train)
        if i==num_iterations-1 :
          loss1_list.append(los1)
          acc1_list.append(acc1)
          sens1=np.linalg.norm(grads_vals1)
          sens_list.append(sens1)
        
        session.run(optimizer2, feed_dict=feed_dict_train)
        los2, acc2 = session.run([loss2, accuracy2], feed_dict=feed_dict_train)
        grads_vals2 = session.run([grad2], feed_dict=feed_dict_train)
        if i==num_iterations-1 :
          loss1_list.append(los2)
          acc1_list.append(acc2)
          sens2=np.linalg.norm(grads_vals2)
          sens_list.append(sens2)
        
        session.run(optimizer3, feed_dict=feed_dict_train)
        los1, acc1 = session.run([loss3, accuracy3], feed_dict=feed_dict_train)
        grads_vals3 = session.run([grad3], feed_dict=feed_dict_train)
        if i==num_iterations-1 :
          loss1_list.append(los1)
          acc1_list.append(acc1)
          sens3=np.linalg.norm(grads_vals3)
          sens_list.append(sens3)

        session.run(optimizer4, feed_dict=feed_dict_train)
        los1, acc1 = session.run([loss4, accuracy4], feed_dict=feed_dict_train)
        grads_vals4 = session.run([grad4], feed_dict=feed_dict_train)
        if i==num_iterations-1 :
          loss1_list.append(los1)
          acc1_list.append(acc1)
          sens4=np.linalg.norm(grads_vals4)
          sens_list.append(sens4)        

        session.run(optimizer5, feed_dict=feed_dict_train)
        los1, acc1 = session.run([loss5, accuracy5], feed_dict=feed_dict_train)
        grads_vals5 = session.run([grad5], feed_dict=feed_dict_train)
        msg = "Iteration1: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}"
        print(msg.format(i + 1, los1, acc1))
        if i==num_iterations-1 :
          loss1_list.append(los1)
          acc1_list.append(acc1)
          sens5=np.linalg.norm(grads_vals5)
          sens_list.append(sens5)
    total_iterations += num_iterations


session = tf.Session()
session.run(tf.global_variables_initializer())

optimize(num_iterations=433)
print(loss1_list)
num=[1e-5,1e-4,1e-3,1e-2,1e-1]
x = np.arange(0, len(num))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(loss1_list,'b--', label = 'loss')
ax2 = ax.twinx()
ax2.plot(sens_list,'g--', label = 'sens')
ax.legend(loc=0)
ax.grid()
ax.set_xlabel("learning rate")
ax.set_ylabel("loss")
ax2.set_ylabel("sens")
#ax2.set_ylim(0.2,0.5)
#ax.set_ylim(2.1, 2.5)
ax2.legend(loc=0)
plt.xticks(x, num)

plt.savefig('1.jpg')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(acc1_list,'r--', label = 'acc')
ax2 = ax.twinx()
ax2.plot(sens_list,'g--', label = 'sens')
ax.legend(loc=0)
ax.grid()
ax.set_xlabel("learning rate")
ax.set_ylabel("acc")
ax2.set_ylabel("sens")
#ax2.set_ylim(0.2,0.5)
#ax.set_ylim(2.1, 2.5)
ax2.legend(loc=0)
plt.xticks(x, num)

plt.savefig('2.jpg')
plt.show()




