import numpy as np
import time

from keras.layers import Dense, Activation
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

N = 500
X, y = datasets.make_moons(N, noise=0.3)
Y = y.reshape(N, 1)

X_train, X_test, Y_train, Y_test =\
    train_test_split(X, Y, train_size=0.8)


print(X)
print('--------------')
print(y)


# plt.plot(X, 'o')
# plt.show()



"""

텐서플로우용 학습 실행기.
 
"""

import tensorflow as tf

num_hidden = 3

x = tf.placeholder(tf.float32, shape=[None, 2])
t = tf.placeholder(tf.float32, shape=[None, 1])

# input ~ hidden
W = tf.Variable(tf.truncated_normal([2, num_hidden]))
b = tf.Variable(tf.zeros([num_hidden]))
h = tf.nn.sigmoid(tf.matmul(x, W) + b)

# hidden ~ output
V = tf.Variable(tf.truncated_normal([num_hidden, 1]))
c = tf.Variable(tf.zeros(1))
y = tf.nn.sigmoid(tf.matmul(h, V) + c)

cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Learning!!

batch_size = 20
n_batches = N #300

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

start_time = time.time()
for epoch in range(500):
    X_, Y_ = shuffle(X_train, Y_train)

    # print('Learning..', epoch, '/ 500')
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size

        sess.run(train_step, feed_dict= {
            x: X_[start:end],
            t: Y_[start:end],
        })

        # print('X_size', len(X_), start, '~', end)\

        if end == len(X_):
            break

print("--- %s seconds ---" %(time.time() - start_time))

accuracy_rate = accuracy.eval(session=sess, feed_dict={
    x: X_test,
    t: Y_test,
})


print('Tensorflow accuracy:', accuracy_rate)


"""

Keras 용

"""
from keras import Sequential
model = Sequential()
model.add(Dense(3, input_dim=2))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=0.05),
              metrics=['accuracy'])

# 학습.
start_time = time.time()
model.fit(X_train, Y_train, epochs=500, batch_size=20)
print("--- %s seconds ---" %(time.time() - start_time))
# 결과
print('-- 결과확인 --')
loss_and_metrics = model.evaluate(X_test, Y_test)
print(loss_and_metrics)


