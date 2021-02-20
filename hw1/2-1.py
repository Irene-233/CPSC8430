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

images = data.test.images[0:9]
cls_true = data.test.cls[0:9]


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
weights_conv1 = get_weights_variable(layer_name='layer1_conv1')
weights_conv2 = get_weights_variable(layer_name='layer1_conv2')
print(weights_conv1)
print(weights_conv2)
weights_fc1 = get_weights_variable(layer_name='layer1_fc')
weights_fc_out = get_weights_variable(layer_name='layer1_fc_out')
print(weights_fc1)
print(weights_fc_out)

session = tf.Session()
session.run(tf.global_variables_initializer())
grads = tf.gradients(loss, weights_fc_out)[0]



total_iterations = 0
train_batch_size = 64
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
        if i % 860 == 0:
            los1, acc1 = session.run([loss1, accuracy1], feed_dict=feed_dict_train)
            loss1_list.append(los1)
            acc1_list.append(acc1)
            msg = "Iteration1: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}"
            print(msg.format(i + 1, los1, acc1))
    total_iterations += num_iterations


optimize(num_iterations=6880)

def plot_conv_weights(weights, input_channel=0):
    w = session.run(weights)
    w_min = np.min(w)
    w_max = np.max(w)
    num_filters = w.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        if i<num_filters:
            img = w[:, :, input_channel, i]
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig('conv1.jpg')
    plt.show()

from sklearn.decomposition import PCA
def plot_fc_weights(weights_list):
    w_list = session.run(weights_list)
    pca = PCA(n_components=2)
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    for w in w_list:      
        print(w.shape)
        principalComponents = pca.fit_transform(w)
        ax.scatter(principalComponents[:,0], principalComponents[:,1], label=w.shape, alpha=0.5)
    ax.legend()
    plt.savefig('whole.jpg')
    plt.show()

plot_conv_weights(weights=weights_conv1)
plot_fc_weights(weights_list=[weights_fc1, weights_fc_out])



