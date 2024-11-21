import tensorflow as tf
# tf.enable_eager_execution()
# import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb
import problems
import copy
import random


np.random.seed(123)
tf.random.set_random_seed(123)
# tf.random.set_random_seed(123)

debug_mode = False

class pso:
    def __init__(
        self,
        fitness_fn,
        mode='test',
        pop_size=10,
        dim=2,
        n_iter=200,
        b=0.9,
        c1=0.8,
        c2=0.5,
        x_min=-1,
        x_max=1,
        x_init = None
        ):
        self.fitness_fn = fitness_fn
        self.mode = mode
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
        self.f_p = self.fitness_fn(self.x,batch_size=self.pop_size,dim=self.dim,mode=self.mode)
        self.fit_history = []
        self.x_history = []
        self.x_history.append(self.x)
        if debug_mode:
            pdb.set_trace()
        self.g = self.p[tf.math.argmin(input=self.f_p)]
        self.v = self.start_velocities()


    def build_swarm(self,x_init):
        """Creates the swarm following the selected initialization method. 
        Returns:
            tf.Tensor: The PSO swarm population. Each particle represents a neural
            network. 
        """
        if x_init!=None:
            # x_internal = tf.Variable(tf.zeros(tf.shape(x_init)), dtype=tf.float32)
            # x_internal.assign(x_init)
            return x_init
        return tf.Variable(
            tf.random.uniform([self.pop_size, self.dim], self.x_min, self.x_max)
        )


    def start_velocities(self):
        """Start the velocities of each particle in the population (swarm). 
        Returns:
            tf.Tensor: The starting velocities.  
        """
        return tf.Variable(
            tf.random.uniform(
                [self.pop_size, self.dim],
                 self.x_min,
                self.x_max ,
            )
        )


    def get_randoms(self):
        """Generate random values to update the particles' positions. 
        Returns:
            _type_: _description_
        """
        return np.random.uniform(0, 1, [2, self.dim])[:, None]

    def update_p_best(self):
        """Updates the *p-best* positions. 
        """
        f_x = self.fitness_fn(self.x,batch_size=self.pop_size,dim=self.dim,mode=self.mode)
        if debug_mode is True:
            pdb.set_trace()
        self.fit_history.append(tf.reduce_mean(f_x))
        
        # f_x_tiled = tf.tile(f_x, [1, self.dim])
        # f_p_tiled = tf.tile(self.f_p, [1, self.dim])
        
        self.p = tf.where(f_x < self.f_p, self.x, self.p)
        self.f_p = tf.where(f_x < self.f_p, f_x, self.f_p)

    def update_g_best(self):
        """Update the *g-best* position. 
        """
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
        self.update_p_best()
        self.update_g_best()

    def train(self):
        """The particle swarm optimization. The PSO will optimize the weights according to the losses of the neural network, so this process is actually the neural network training. 
        """
        for i in range(self.n_iter):
            self.step()
            self.x_history.append(self.x)

def objective_function(X):
    return tf.math.sqrt(X[:,0]**2 + X[:,1]**2)[:,None]

def rastrigin_function(X, dim=2):
    A = 10
    return A * dim + tf.reduce_sum(tf.square(X) - A * tf.cos(2 * np.pi * X), axis=1, keepdims=True)

def fitness_function():
    def f(X,batch_size,dim,mode):
        return square_cos(X,batch_size=batch_size,num_dims=dim,mode=mode)
    return f

def square_cos(x,mode='train',batch_size=10,num_dims=5,dtype=tf.float32,stddev=0.01):
    # Trainable variable.
    if mode=='test':
    #   x = x - 1.0
      return ( tf.reduce_sum(x*x - 10*tf.math.cos(2*3.1415926*x), 1)+ 10*num_dims )
    # x = x - 1.0
    product = tf.squeeze(tf.matmul(w, tf.expand_dims(x, -1)))
    product2 = tf.reduce_sum(wcos*10*tf.math.cos(2*3.1415926*x), 1)
    #product3 = tf.reduce_sum((product - y) ** 2, 1) - tf.reduce_sum(product2, 1) + 10*num_dims
    return (tf.reduce_sum((product - y) ** 2, 1)) - tf.reduce_mean(product2) + 10*num_dims

def quadratic(x,mode='train',batch_size=10,num_dims=5,dtype=tf.float32,stddev=0.01):
    """Quadratic problem: f(x) = ||Wx - y||. Builds loss graph."""
    with tf.variable_scope('PSO', reuse=tf.AUTO_REUSE):
        w = tf.get_variable("w",
                            shape=[batch_size, num_dims, num_dims],
                            dtype=dtype,
                            initializer=tf.random_uniform_initializer(),
                            trainable=False)
        y = tf.get_variable("y",
                            shape=[batch_size, num_dims],
                            dtype=dtype,
                            initializer=tf.random_uniform_initializer(),
                            trainable=False)
        product = tf.squeeze(tf.matmul(w, tf.expand_dims(x, -1)))
        return (tf.reduce_sum((product - y) ** 2, 1))
    
def rosenbrock(x, batch_size=10, num_dims=2, dtype=tf.float32, mode='test'):
    """Builds loss graph."""
    # Trainable variable.
    # x = tf.get_variable(
    #     "x",
    #     shape=[batch_size, num_dims],
    #     dtype=dtype,
    #     initializer=tf.random_uniform_initializer(-3, 3)
    # )

    if mode=='test':
        a = tf.constant(1, dtype=dtype)
        b = tf.constant(100, dtype=dtype)
        x1 = x[:, :-1]
        x2 = x[:, 1:]
        return tf.reduce_sum(b * (x2 - x1**2.0)**2.0 + (a - x1)**2.0, axis=1)

    # In training mode, 'a' and 'b' are randomly initialized
    a = tf.get_variable("a",
                        shape=[batch_size, num_dims-1],
                        dtype=dtype,
                        initializer=tf.random_uniform_initializer(-3, 3),
                        trainable=False)

    b = tf.get_variable("b",
                        shape=[batch_size, num_dims-1],
                        dtype=dtype,
                        initializer=tf.random_uniform_initializer(80, 120),
                        trainable=False)

    # Compute Rosenbrock function
    x1 = x[:, :-1]
    x2 = x[:, 1:]
    rosenbrock_value = tf.reduce_sum(b * (x2 - x1**2.0)**2.0 + (a - x1)**2.0, axis=1)
    return rosenbrock_value

def weierstrass(x, batch_size=128, num_dims=2, dtype=tf.float32, mode='train'):
    """Weierstrass function problem: f(x)."""
        
    x = 0.005 * x
    k = tf.range(start=0, limit=21, delta=1, dtype=tf.float32)
    k = tf.expand_dims(tf.expand_dims(k, 0), 0)
    ak = 0.5**k
    bk = tf.constant(np.pi) * (3**k)

    # Different settings for training and testing
    if mode=='test':
        # Compute Weierstrass function
        kcs = ak * tf.cos(2*(tf.expand_dims(x, -1) + 0.5)*bk)  # shape (M, nx, 21)
    else:
        # In training mode, 'a' and 'b' are randomly initialized
        ak = tf.random_uniform([batch_size, 1, 21], minval=0, maxval=1, dtype=dtype)
        bk = tf.random_uniform([batch_size, 1, 21], minval=np.pi, maxval=3*np.pi, dtype=dtype)
        kcs = ak * tf.cos(2*(tf.expand_dims(x, -1) + tf.random.uniform([batch_size, num_dims, 1], minval=0, maxval=1, dtype=dtype))*bk)

    ksm = tf.reduce_sum(kcs, axis=2)
    sm = tf.reduce_sum(ksm, axis=1)

    kcs = ak * tf.cos(bk)
    ksm = tf.reduce_sum(kcs)

    return sm - tf.cast(tf.shape(x)[1], tf.float32)*ksm

def smoothed_weierstrass_tf(x, a=0.5, b=3, N=50, sigma=0.5):
    total = tf.constant(0.0, dtype=tf.float32)
    for n in range(N):
        total += a**n * tf.cos(b**n * 3.1415926 * x)
    return total * tf.exp(-sigma**2 * x**2)

def weierstrass_smoothed(batch_size=128, num_dims=None, stddev=0.01, dtype=tf.float32, mode='train'):
    print(num_dims)

    """Builds loss graph."""

    # Trainable variable.
    if mode=='test':
        # x = tf.get_variable(
        #     "x",
        #     shape=[batch_size, num_dims],
        #     dtype=dtype,
        #     initializer=tf.random_uniform_initializer(-3, 3))
        return tf.reduce_sum(smoothed_weierstrass_tf(x))

    # x = tf.get_variable(
    #     "x",
    #     shape=[batch_size, num_dims],
    #     dtype=dtype,
    #     initializer=tf.random_uniform_initializer(-3, 3))

    # Non-trainable variables.
    w = tf.get_variable("w",
                        dtype=dtype,
                        initializer=indentity_init(batch_size, num_dims, stddev/num_dims),
                        trainable=False)

    y = tf.get_variable("y",
                        shape=[batch_size, num_dims],
                        dtype=dtype,
                        initializer=tf.random_normal_initializer(stddev=stddev/num_dims),
                        trainable=False)

    product = tf.squeeze(tf.matmul(w, tf.expand_dims(x, -1)))

    weierstrass_terms = tf.reduce_sum(smoothed_weierstrass_tf(x), 1)

    return tf.reduce_sum((product - y) ** 2, 1) - tf.reduce_mean(weierstrass_terms)


def ackley(x, batch_size=128, num_dims=10, dtype=tf.float32, mode='train'):
    """Ackley problem with varying parameters for training."""
    
    def ackley_tf(x, a, b, c):
        smsq = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
        smcs = tf.reduce_sum(tf.cos(c * x), axis=1, keepdims=True)
        inx = 1.0 / tf.cast(tf.shape(x)[1], dtype)
        term1 = -a * tf.exp(-b * tf.sqrt(inx * smsq))
        term2 = -tf.exp(inx * smcs)
        return tf.squeeze(term1 + term2 + a + tf.constant(np.e, dtype=dtype))
    

    # """Builds loss graph."""
    # x = tf.get_variable(
    #     "x",
    #     shape=[batch_size, num_dims],
    #     dtype=dtype,
    #     initializer=tf.random_normal_initializer(stddev=0.01))
    
    if mode == 'test':
        # Fixed parameters for testing
        a = tf.constant(20.0, shape=[batch_size, 1], dtype=dtype)
        b = tf.constant(0.2, shape=[batch_size, 1], dtype=dtype)
        c = tf.constant(2.0 * np.pi, shape=[batch_size, 1], dtype=dtype)
    else:
        # Random parameters for training
        a = tf.random_uniform([batch_size, 1], minval=18, maxval=22, dtype=dtype)
        b = tf.random_uniform([batch_size, 1], minval=0.18, maxval=0.22, dtype=dtype)
        c = tf.random_uniform([batch_size, 1], minval=2.0*np.pi-0.1, maxval=2.0*np.pi+0.1, dtype=dtype)

    # Compute Ackley function
    ackley_value = ackley_tf(x, a, b, c)
    
    return ackley_value



if __name__ == '__main__':
    pop_size = 7
    dim = 5
    n_iter = 250
    n_epochs = 100
    x_val = tf.get_variable("x",shape=[pop_size,dim],dtype=np.float32,initializer=tf.random_uniform_initializer(-3, 3))
    # w = tf.get_variable("w",dtype=np.float32,initializer=problems.indentity_init(1, 2, 0.01/2),trainable=False)
    w = tf.get_variable("w",
                    shape=[1, dim, dim],
                    dtype=np.float32,
                    initializer=tf.random_uniform_initializer(),
                    # initializer=tf.constant_initializer(1.0),
                    trainable=False)
    # w = tf.get_variable("w", dtype=np.float32,initializer=tf.constant_initializer(1.0),trainable=False)
    y = tf.get_variable("y",shape=[pop_size, dim],dtype=np.float32,initializer=tf.random_normal_initializer(stddev=0.01/2),trainable=False)
    # y = tf.get_variable("y",shape=[pop_size, dim],dtype=np.float32,initializer=tf.constant_initializer(-1.0),trainable=False)
    wcos = tf.get_variable("wcos",shape=[pop_size, dim],dtype=np.float32,initializer=tf.random_normal_initializer(mean=1.0, stddev=0.01/2),trainable=False)
    with tf.Session() as sess:
        for i in range(n_epochs):
        
            print('Epoch ', i)
            seed_value = random.randint(0, 100)
            tf.set_random_seed(seed_value)

        
            
            opt = pso(fitness_fn=fitness_function(),pop_size=pop_size, dim=dim, n_iter=n_iter,x_init=x_val)
            sess.run(tf.global_variables_initializer())
            opt.train()
            # print(sess.run(opt.fit_history))
            print('final cost:', min(sess.run(opt.fit_history)))
            # print(sess.run(opt.x))
            # print(sess.run(opt.x_history))
            # print(opt.f_p)
        # print(opt.fit_history)
        ## Define the grid for future plotting:
        
        
        # anim = animation.FuncAnimation(fig,snapshot,frames=60)
        # anim.save("PSO_tensorflow.mp4", fps=6)