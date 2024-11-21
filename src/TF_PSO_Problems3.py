import tensorflow as tf
#tf.enable_eager_execution()
# import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import problems
import pdb


np.random.seed(123)
tf.random.set_random_seed(123)
# tf.random.set_random_seed(123)

debug_mode = False

class pso:
    def __init__(
        self,
        session,
        fitness_fn,
        pop_size=10,
        dim=2,
        n_iter=200,
        b=0.9,
        c1=0.8,
        c2=0.5,
        x_min=-1,
        x_max=1,
        x_init=None,
        v_init=None
        ):
        self.sess = session
        self.fitness_fn = fitness_fn
        self.pop_size = pop_size
        self.dim = dim
        self.n_iter = n_iter
        self.b = b
        self.c1 = c1
        self.c2 = c2
        self.x_min = x_min
        self.x_max = x_max
        self.x = self.build_swarm(x_init)
        self.p = self.x
        self.f_p = self.fitness_fn(self.sess,self.x)
        # self.f_p = self.fitness_fn
        self.fit_history = []
        if debug_mode is True:
            pdb.set_trace()
        self.g = self.p[tf.math.argmin(input=self.f_p)]
        # self.g = self.p[tf.math.argmin(input=self.f_p).numpy()[0]]
        # self.g = self.p[tf.math.argmin(input=tf.reshape(self.f_p,[self.pop_size,1])).numpy()[0]]
        self.v = self.start_velocities(v_init)


def build_swarm(self, x_init=None):
    if x_init is None:
        x_init = tf.random.uniform([self.pop_size, self.dim], self.x_min, self.x_max)
    x = tf.Variable(x_init)
    return x

def start_velocities(self, v_init=None):
    if v_init is None:
        return tf.Variable(tf.random.uniform([self.pop_size, self.dim], self.x_min, self.x_max))
    return v_init

def get_randoms(self):
    return np.random.uniform(0, 1, [2, self.dim])[:, None]

def step(self):
    r1, r2 = self.get_randoms()
    v = (
        self.b * self.v
        + self.c1 * r1 * (self.p - self.x)
        + self.c2 * r2 * (self.g - self.x)
    )
    x = tf.clip_by_value(self.x + v, self.x_min, self.x_max)

    def update_p_best():
        f_x = self.fitness_fn(self.sess, x)
        self.p = tf.cond(f_x < self.f_p, lambda: x, lambda: self.p)
        self.f_p = tf.cond(f_x < self.f_p, lambda: f_x, lambda: self.f_p)
        return f_x

    def update_g_best():
        self.g = tf.gather(self.p, tf.math.argmin(input=self.f_p))

    f_x = tf.cond(
        tf.equal(tf.shape(self.fit_history)[0], 0),
        lambda: update_p_best(),
        lambda: tf.cond(
            tf.less(tf.random.uniform([]), 0.5),
            lambda: self.fitness_fn(self.sess, x),
            lambda: update_p_best(),
        ),
    )

    self.fit_history = tf.concat([self.fit_history, [tf.reduce_mean(f_x)]], axis=0)
    update_g_best()

    return tf.group([self.x.assign(x), self.v.assign(v)])

@tf.function
def train(self):
    i = tf.constant(0)
    cond = lambda i, _: tf.less(i, self.n_iter)
    body = lambda i, _: [tf.add(i, 1), self.step()]
    tf.while_loop(cond, body, loop_vars=[i, self.get_train_iterator()])

# with tf.variable_scope("square_cos", reuse=tf.AUTO_REUSE):
problem =  problems.square_cos(batch_size=10, num_dims=2,  stddev=0.01, dtype=tf.float32, mode='test')()
def special_objective_function(prob):
    def f(sess,value):
        x = tf.get_variable("x", shape=[10, 2], dtype=tf.float32)
        # output = sess.run(f, feed_dict={"square_cos/x:0": value})
        sess.run(tf.assign(x, value))
        output = sess.run(prob)
        return output
    return f

                                 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    x_val = tf.random_uniform([10, 2], minval=-3, maxval=3,dtype=tf.float32)
    v_val = tf.random_uniform([10, 2], minval=-0.1, maxval=0.1,dtype=tf.float32)
    pso_solver = pso(
        sess,
        special_objective_function(problem),
        pop_size=10,
        dim=2,
        n_iter=100,
        b=0.9,
        c1=0.8,
        c2=0.5,
        x_min=-3,
        x_max=3,
        x_init=x_val,
        v_init=v_val
        )
    pso_solver.train()

    plt.plot(pso_solver.fit_history)
    plt.title("Objective function over time")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function")
    plt.show()
