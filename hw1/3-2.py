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
logits3 = tf.layers.dense(inputs=net3, name='layer_fc_out',
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

# model six #
net6 = tf.layers.conv2d(inputs=x_image, name='layer6_conv1', padding='same',
                       filters=16, kernel_size=5, activation=tf.nn.relu)
net6 = tf.layers.max_pooling2d(inputs=net6, pool_size=2, strides=2)
net6 = tf.layers.conv2d(inputs=net6, name='layer6_conv2', padding='same',
                       filters=36, kernel_size=5, activation=tf.nn.relu)
net6 = tf.layers.max_pooling2d(inputs=net6, pool_size=2, strides=2)

net6 = tf.layers.flatten(net6)
net6 = tf.layers.dense(inputs=net6, name='layer6_fc',
                      units=2048, activation=tf.nn.relu)
                                                                
logits6 = tf.layers.dense(inputs=net6, name='layer6_fc_out',
                      units=num_classes, activation=None)
y6_pred = tf.nn.softmax(logits=logits6)
y6_pred_cls = tf.argmax(y6_pred, dimension=1)

cross_entropy6 = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits6)
loss6 = tf.reduce_mean(cross_entropy6)
opt6 = tf.train.AdamOptimizer(learning_rate=1e-4)
optimizer6 = opt1.minimize(loss6)
correct_prediction6 = tf.equal(y6_pred_cls, y_true_cls)
accuracy6 = tf.reduce_mean(tf.cast(correct_prediction6, tf.float32))

# model seven #
net7 = tf.layers.conv2d(inputs=x_image, name='layer7_conv1', padding='same',
                       filters=16, kernel_size=5, activation=tf.nn.relu)
net7 = tf.layers.max_pooling2d(inputs=net7, pool_size=2, strides=2)
net7 = tf.layers.conv2d(inputs=net7, name='layer7_conv2', padding='same',
                       filters=36, kernel_size=5, activation=tf.nn.relu)
net7 = tf.layers.max_pooling2d(inputs=net7, pool_size=2, strides=2)

net7 = tf.layers.flatten(net7)
net7 = tf.layers.dense(inputs=net7, name='layer7_fc',
                      units=4096, activation=tf.nn.relu)
                                                                                       
logits7 = tf.layers.dense(inputs=net7, name='layer7_fc_out',
                      units=num_classes, activation=None)
y7_pred = tf.nn.softmax(logits=logits7)
y7_pred_cls = tf.argmax(y7_pred, dimension=1)

cross_entropy7 = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits7)
loss7 = tf.reduce_mean(cross_entropy7)
opt7 = tf.train.AdamOptimizer(learning_rate=1e-4)
optimizer7 = opt1.minimize(loss7)
correct_prediction7 = tf.equal(y7_pred_cls, y_true_cls)
accuracy7 = tf.reduce_mean(tf.cast(correct_prediction7, tf.float32))

#model eight #
net8 = tf.layers.conv2d(inputs=x_image, name='layer8_conv1', padding='same',
                       filters=16, kernel_size=5, activation=tf.nn.relu)
net8 = tf.layers.max_pooling2d(inputs=net8, pool_size=2, strides=2)
net8 = tf.layers.conv2d(inputs=net8, name='layer8_conv2', padding='same',
                       filters=36, kernel_size=5, activation=tf.nn.relu)
net8 = tf.layers.max_pooling2d(inputs=net8, pool_size=2, strides=2)

net8 = tf.layers.flatten(net8)
net8 = tf.layers.dense(inputs=net8, name='layer8_fc',
                      units=8192, activation=tf.nn.relu)

logits8 = tf.layers.dense(inputs=net8, name='layer8_fc_out',
                      units=num_classes, activation=None)
y8_pred = tf.nn.softmax(logits=logits8)
y8_pred_cls = tf.argmax(y8_pred, dimension=1)

cross_entropy8 = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits8)
loss8 = tf.reduce_mean(cross_entropy8)
opt8 = tf.train.AdamOptimizer(learning_rate=1e-4)
optimizer8 = opt1.minimize(loss8)
correct_prediction8 = tf.equal(y8_pred_cls, y_true_cls)
accuracy8 = tf.reduce_mean(tf.cast(correct_prediction8, tf.float32))

# model nine #
net9 = tf.layers.conv2d(inputs=x_image, name='layer9_conv1', padding='same',
                       filters=16, kernel_size=5, activation=tf.nn.relu)
net9 = tf.layers.max_pooling2d(inputs=net9, pool_size=2, strides=2)
net9 = tf.layers.conv2d(inputs=net9, name='layer9_conv2', padding='same',
                       filters=36, kernel_size=5, activation=tf.nn.relu)
net9 = tf.layers.max_pooling2d(inputs=net9, pool_size=2, strides=2)

net9 = tf.layers.flatten(net2)
net9 = tf.layers.dense(inputs=net9, name='layer9_fc',
                      units=16384, activation=tf.nn.relu)

logits9 = tf.layers.dense(inputs=net9, name='layer9_fc_out',
                      units=num_classes, activation=None)
y9_pred = tf.nn.softmax(logits=logits9)
y9_pred_cls = tf.argmax(y9_pred, dimension=1)

cross_entropy9 = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits9)
loss9 = tf.reduce_mean(cross_entropy9)
opt9 = tf.train.AdamOptimizer(learning_rate=1e-4)
optimizer9 = opt1.minimize(loss9)
correct_prediction9 = tf.equal(y9_pred_cls, y_true_cls)
accuracy9 = tf.reduce_mean(tf.cast(correct_prediction9, tf.float32))

#model ten# 
net10 = tf.layers.conv2d(inputs=x_image, name='layer10_conv1', padding='same',
                       filters=16, kernel_size=5, activation=tf.nn.relu)
net10 = tf.layers.max_pooling2d(inputs=net10, pool_size=2, strides=2)
net10 = tf.layers.conv2d(inputs=net10, name='layer10_conv2', padding='same',
                       filters=36, kernel_size=5, activation=tf.nn.relu)
net10 = tf.layers.max_pooling2d(inputs=net10, pool_size=2, strides=2)

net10 = tf.layers.flatten(net10)
net10 = tf.layers.dense(inputs=net10, name='layer10_fc',
                      units=32768, activation=tf.nn.relu)


logits10 = tf.layers.dense(inputs=net10, name='layer10_fc_out',
                      units=num_classes, activation=None)
y10_pred = tf.nn.softmax(logits=logits10)
y10_pred_cls = tf.argmax(y10_pred, dimension=1)

cross_entropy10 = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits10)
loss10 = tf.reduce_mean(cross_entropy10)
opt10 = tf.train.AdamOptimizer(learning_rate=1e-4)
optimizer10 = opt1.minimize(loss10)
correct_prediction10 = tf.equal(y10_pred_cls, y_true_cls)
accuracy10 = tf.reduce_mean(tf.cast(correct_prediction10, tf.float32))

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
        los1, acc1 = session.run([loss1, accuracy1], feed_dict=feed_dict_train)
        if i==num_iterations-1 :
          loss1_list.append(los1)
          acc1_list.append(acc1)
        
        session.run(optimizer2, feed_dict=feed_dict_train)
        los2, acc2 = session.run([loss2, accuracy2], feed_dict=feed_dict_train)
        if i==num_iterations-1 :
          loss1_list.append(los2)
          acc1_list.append(acc2)
        
        session.run(optimizer3, feed_dict=feed_dict_train)
        los1, acc1 = session.run([loss3, accuracy3], feed_dict=feed_dict_train)
        if i==num_iterations-1 :
          loss1_list.append(los1)
          acc1_list.append(acc1)

        session.run(optimizer4, feed_dict=feed_dict_train)
        los1, acc1 = session.run([loss4, accuracy4], feed_dict=feed_dict_train)
        if i==num_iterations-1 :
          loss1_list.append(los1)
          acc1_list.append(acc1)        

        session.run(optimizer5, feed_dict=feed_dict_train)
        los1, acc1 = session.run([loss5, accuracy5], feed_dict=feed_dict_train)
        if i==num_iterations-1 :
          loss1_list.append(los1)
          acc1_list.append(acc1)

        session.run(optimizer6, feed_dict=feed_dict_train)
        los1, acc1 = session.run([loss6, accuracy6], feed_dict=feed_dict_train)
        if i==num_iterations-1 :
          loss1_list.append(los1)
          acc1_list.append(acc1)

        session.run(optimizer7, feed_dict=feed_dict_train)
        los1, acc1 = session.run([loss7, accuracy7], feed_dict=feed_dict_train)
        if i==num_iterations-1 :
          loss1_list.append(los1)
          acc1_list.append(acc1)

        session.run(optimizer8, feed_dict=feed_dict_train)
        los1, acc1 = session.run([loss8, accuracy8], feed_dict=feed_dict_train)
        if i==num_iterations-1 :
          loss1_list.append(los1)
          acc1_list.append(acc1)
        
        session.run(optimizer9, feed_dict=feed_dict_train)
        los1, acc1 = session.run([loss9, accuracy9], feed_dict=feed_dict_train)
        if i==num_iterations-1 :
          loss1_list.append(los1)
          acc1_list.append(acc1)

        session.run(optimizer10, feed_dict=feed_dict_train)
        los1, acc1 = session.run([loss10, accuracy10], feed_dict=feed_dict_train)
        if i==num_iterations-1 :
          loss1_list.append(los1)
          acc1_list.append(acc1)
    total_iterations += num_iterations


session = tf.Session()
session.run(tf.global_variables_initializer())

optimize(num_iterations=500)
print(loss1_list)
num=[64,128,256,512,1024,2048,4096,8192,16384,32768]
x = np.arange(0, len(num))
plt.plot(loss1_list,'bo',loss2_list,'r*')
plt.xticks(x, num)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('number of parameters')
plt.savefig('loss.jpg')
plt.show()
plt.plot(acc1_list,'bo',acc2_list,'r*')
plt.xticks(x, num)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('number of parameters')
plt.savefig('acc.jpg')
plt.show()






