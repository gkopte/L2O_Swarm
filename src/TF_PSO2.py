import tensorflow as tf
tf.enable_eager_execution()
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


    def build_swarm(self,x_init=None):
        """Creates the swarm following the selected initialization method. 
        Returns:
            tf.Tensor: The PSO swarm population. Each particle represents a neural
            network.
        """
        if x_init is None:
            return tf.Variable(
                tf.random.uniform([self.pop_size, self.dim], self.x_min, self.x_max)
            )
        # x = tf.get_variable(
        # "x",
        # shape=[self.pop_size, self.dim],
        # dtype=tf.float32,
        # initializer=tf.random_uniform_initializer(-3, 3))
        # x = tf.get_default_graph().get_tensor_by_name("my_scope/x:0")
        # x = tf.get_default_graph().get_tensor_by_name("x")
        # x = tf.get_variable("square_cos/x0", shape=[10, 2], dtype=tf.float32)

        return x_init


    def start_velocities(self,v):
        """Start the velocities of each particle in the population (swarm). 
        Returns:
            tf.Tensor: The starting velocities.  
        """
        return v
        # return tf.Variable(
        #     tf.random.uniform(
        #         [self.pop_size, self.dim],
        #          self.x_min,
        #         self.x_max ,
        #     )
        # )


    def get_randoms(self):
        """Generate random values to update the particles' positions. 
        Returns:
            _type_: _description_
        """
        return np.random.uniform(0, 1, [2, self.dim])[:, None]

    def update_p_best(self):
        """Updates the *p-best* positions. 
        """
        # f_x = self.fitness_fn(self.x)
        f_x = self.fitness_fn(sess,self.x)
        # self.fit_history.append(tf.reduce_mean(f_x).numpy())
        self.fit_history.append(tf.reduce_mean(f_x))
        if debug_mode is True:
            pdb.set_trace()
        # f_x_tiled = tf.tile(f_x, [1, self.dim])
        # replication_vector = tf.constant([1, self.dim])
        # f_x_tiled = tf.tile(f_x,replication_vector)
        # f_p_tiled = tf.tile(self.f_p, [1, self.dim])
        
        self.p = tf.where(f_x < self.f_p, self.x, self.p)
        self.f_p = tf.where(f_x < self.f_p, f_x, self.f_p)

    def update_g_best(self):
        """Update the *g-best* position. 
        """
        if debug_mode is True and False:
            pdb.set_trace()
        # self.g = self.p[tf.math.argmin(input=self.f_p).numpy()[0]]
        self.g = self.p[tf.math.argmin(input=self.f_p)]


    def step(self):
        """It runs ONE step on the particle swarm optimization. 
        """
        r1, r2 = self.get_randoms()
        self.v = (
            self.b * self.v
            + self.c1 * r1 * (self.p - self.x)
            + self.c2 * r2 * (self.g - self.x)
        )
        self.x = tf.clip_by_value(self.x + self.v, self.x_min, self.x_max)
        if debug_mode is True:
            pdb.set_trace()

        # x = tf.get_variable(
        # "x",
        # shape=[self.pop_size, self.dim],
        # dtype=tf.float32,
        # initializer=tf.random_uniform_initializer(-3, 3))
    
        # sess.run(self.fitness_fn, feed_dict={"my_scope/x:0": self.x})


        self.update_p_best()
        self.update_g_best()

    def train(self):
        """The particle swarm optimization. The PSO will optimize the weights according to the losses of the neural network, so this process is actually the neural network training. 
        """
        for i in range(self.n_iter):
            self.step()

def objective_function(X):
    return tf.math.sqrt(X[:,0]**2 + X[:,1]**2)[:,None]

def rastrigin_function(X, dim=2):
    A = 10
    return A * dim + tf.reduce_sum(tf.square(X) - A * tf.cos(2 * np.pi * X), axis=1, keepdims=True)

def fitness_function():
    def f(X):
        return objective_function(X)
    return f

with tf.variable_scope("square_cos", reuse=tf.AUTO_REUSE):
    problem =  problems.square_cos(batch_size=10, num_dims=2,  stddev=0.01, dtype=tf.float32, mode='test')
    def special_objective_function(sess,value):
        
            x = tf.get_variable("x", shape=[10, 2], dtype=tf.float32)
            f = problem()
            # output = sess.run(f, feed_dict={"square_cos/x:0": value})
            sess.run(tf.assign(x, value))
            output = sess.run(f)
            return output


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        x_val = tf.random_uniform([10, 2], minval=-3, maxval=3,dtype=tf.float32)
        v_val = tf.zeros([10, 2],dtype=tf.float32)
        # output = special_objective_function(sess,x_val)
        # print(output)

        opt = pso(sess,special_objective_function,pop_size=10, dim=2, n_iter=100,x_min=-3,x_max=3,x_init=x_val,v_init=v_val)
        # opt = pso(fitness_fn=fitness_function(),pop_size=10, dim=2, n_iter=128)
        opt.train()


        print(sess.run(opt.g))
        print(sess.run(opt.f_p))
        print(sess.run(opt.fit_history))
