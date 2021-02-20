import tensorflow.compat.v1 as tf
#import tensorflow as tf
#sess=tf.compat.v1.InteractiveSession()
sess=tf.InteractiveSession()
import numpy as np
X =np.expand_dims(np.arange(0.0, 3.0, 0.01),1)
Y =np.sinc(X)

#x = tf.compat.v1.placeholder(shape=[300,1], dtype=tf.float64)
#y = tf.compat.v1.placeholder(shape=[300,1], dtype=tf.float64)
x = tf.placeholder(tf.float64, [300,1], name='x')
y = tf.placeholder(tf.float64, [300,1], name='y')
input_layer = tf.layers.dense(x, 15, activation= tf.nn.relu)
hidden_layer1 = tf.layers.dropout(input_layer,0.2)
hidden_layer2 = tf.layers.dense(hidden_layer1,300,activation=tf.nn.relu)
output_layer = tf.layers.dense(hidden_layer2,1)
Loss =tf.losses.mean_squared_error(y , output_layer)
Optimizer = tf.train.AdamOptimizer(learning_rate= 0.001).minimize(Loss)
init = tf.global_variables_initializer()
loss_list=[]
sess.run(init)
for i in range(0,1000):
  fd ={x:X, y:Y}
  _, loss_val = sess.run([Optimizer, Loss], feed_dict=fd)
  #print ('loss = %s' % loss_val)
  loss_list.append(loss_val)

YP = sess.run(output_layer,feed_dict={x:X})
# Plot training  loss values
import matplotlib.pyplot as plt
plt.plot(loss_list)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('loss.jpg')
plt.show()
plt.plot(X,Y)
plt.plot(X,YP)
plt.savefig('fuc.jpg')
plt.show()