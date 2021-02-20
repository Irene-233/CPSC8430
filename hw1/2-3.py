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

loss=tf.losses.mean_squared_error(y_true,logits1)

opt1 = tf.train.AdamOptimizer(learning_rate=1e-4)
optimizer1 = opt1.minimize(loss)
optimizer2= opt1.minimize(loss1)

correct_prediction1 = tf.equal(y1_pred_cls, y_true_cls)
accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))

trainable_var_list = tf.trainable_variables()
def get_weights_variable(layer_name):
    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('kernel')
    return variable
weights_conv1 = get_weights_variable(layer_name='layer1_conv1')
weights_conv2 = get_weights_variable(layer_name='layer1_conv2')
weights_fc1 = get_weights_variable(layer_name='layer1_fc')
weights_fc_out = get_weights_variable(layer_name='layer1_fc_out')

session = tf.Session()
session.run(tf.global_variables_initializer())
grads = tf.gradients(loss, weights_fc_out)[0]

hessian = tf.reduce_sum(tf.hessians(loss, weights_fc_out)[0], axis = 2)

total_iterations = 0
train_batch_size = 128
loss1_list=[]
acc1_list=[]
loss2_list=[]
acc2_list=[]
grads_list=[]
mini_grad=10.0
mini_hess=0.0

def optimize(num_iterations):
    global total_iterations
    global mini_grad
    global mini_hess
    for i in range(total_iterations,
                   total_iterations + num_iterations):
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        if i==0:
          session.run(optimizer2, feed_dict=feed_dict_train)
        session.run(optimizer1, feed_dict=feed_dict_train)
        if i % 1 == 0:
            los1, acc1 = session.run([loss, accuracy1], feed_dict=feed_dict_train)
            grads_vals,hess = session.run([grads,hessian], feed_dict=feed_dict_train)

            loss1_list.append(los1)
            acc1_list.append(acc1)
            grad=0.0
            for p in grads_vals:
              thegrads=np.linalg.norm(p,ord=2)
              grad=grad+thegrads 
            grads_list.append(grad)
            if(i>1):
              if(grad<mini_grad):
                mini_grad=grad
                mini_hess=hess
            msg = "Iteration1: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}"
            print(msg.format(i + 1, los1, acc1))
    total_iterations += num_iterations



optimize(num_iterations=12500)
print(mini_grad)
print(mini_hess)
plt.plot(grads_list)
plt.title('Model grads')
plt.ylabel('grads')
plt.xlabel('iteration')
plt.savefig('grads.jpg')
plt.show()
