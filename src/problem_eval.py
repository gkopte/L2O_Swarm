import tensorflow as tf
import problems

num_dims = 2
x_val = tf.constant([[1, 1]], dtype=tf.float32)

with tf.variable_scope("square_cos", reuse=tf.AUTO_REUSE):
    x = tf.get_variable("x", shape=[1, num_dims], dtype=tf.float32)
    square_cos_func = problems.square_cos(batch_size=1, num_dims=num_dims, mode='test')
    result = square_cos_func()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.assign(x, x_val))
    result_val = sess.run(result)
    print("Result: ", result_val)
