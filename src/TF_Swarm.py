import tensorflow as tf
import tensorflow_swarm as swarm

def optimize_function():
    x = tf.Variable(tf.random_uniform([1], -10, 10))
    y = tf.square(x)
    swarm_optimizer = swarm.SwarmOptimizer(10, 0.5, 0.5)
    train_op = swarm_optimizer.minimize(y)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            sess.run(train_op)
        print(sess.run(x))
