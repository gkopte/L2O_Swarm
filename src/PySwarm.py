import pyswarms as ps
import numpy as np
import problems
import tensorflow as tf
#import util

# Set-up the swarm
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
batch_size = 10
num_dims = 2

# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=batch_size, dimensions=num_dims, options=options)

num_dims = 2


def f(x,batch_size=128,num_dims=2):

    # x_val = tf.constant([[1, 2]], dtype=tf.float32)
    #print(x)
    #print(x.shape)
    x_val = tf.constant(x, dtype=tf.float32)
    with tf.variable_scope("square_cos", reuse=tf.AUTO_REUSE):
        square_cos_func = problems.square_cos(batch_size=10,num_dims=2, mode='train')
        result = square_cos_func()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(result)
        return result.tolist()

# print(f(np.array([1,2]),batch_size,batch_size))

# Optimize for a certain number of iterations
cost, pos = optimizer.optimize(f, iters=1000, verbose=False)
print(cost)
print(pos)
