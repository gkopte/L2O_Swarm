import tensorflow as tf
# tf.enable_eager_execution()
# import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import problems
import pdb

np.random.seed(123)
tf.random.set_random_seed(123)
# tf.random.set_random_seed(123)

debug_mode = True

class pso:
    def __init__(
        self,
        session,
        # problem,
        x,
        fitness_fn,
        pop_size=10,
        dim=2,
        n_iter=200,
        b=0.9,
        c1=0.8,
        c2=0.5,
        x_min=-1,
        x_max=1,
        ):
        self.x_ = x
        self.sess = session
        # self.problem = problem
        self.fitness_fn = fitness_fn
        self.pop_size = pop_size
        self.dim = dim
        self.n_iter = n_iter
        self.b = b
        self.c1 = c1
        self.c2 = c2
        self.x_min = x_min
        self.x_max = x_max
        self.x = self.build_swarm()
        self.p = self.x
        self.f_p = self.fitness_fn(x,self.x)
        self.fit_history = []
        if debug_mode is True:
            pdb.set_trace()
        self.g = self.p[tf.math.argmin(input=self.f_p)]
        self.v = self.start_velocities()


    def build_swarm(self):
        """Creates the swarm following the selected initialization method. 
        Returns:
            tf.Tensor: The PSO swarm population. Each particle represents a neural
            network. 
        """
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
        f_x = self.fitness_fn(x,self.x)
        self.fit_history.append(tf.reduce_mean(f_x))
        if debug_mode is True:
            pdb.set_trace()
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
        return sess.run((self.p,self.f_p,self.g))

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


## Define the grid for future plotting:

def fitness_fn(x,x_val):
  #print("x shape: {}".format(x.shape))
#   x_val = np.array(x_val,dtype=np.float32).reshape(1,-1)
  #print("x_val shape: {}".format(x_val.shape))
#   sess.run(x.assign(x_val))
  #print(sess.run(x))
  x.assign(x_val)
  problem = problems.square_cos(batch_size=10, num_dims=2, mode='test')()
  return problem

if __name__ == '__main__':
  with tf.Session() as sess:

    with tf.variable_scope("square_cos", reuse=tf.AUTO_REUSE):
        sess.run(tf.global_variables_initializer())
        x = tf.get_variable("x",shape=[10, 2],dtype=np.float32,initializer=tf.random_uniform_initializer(-3, 3))
        opt = pso(sess,x,fitness_fn=fitness_fn,pop_size=10, dim=2, n_iter=1)
        out = opt.step()
        print(out)


# if __name__ == '__main__':
#   num_particles = 10
#   max_iter = 128
#   with tf.Session() as sess:
    
#     with tf.variable_scope("square_cos", reuse=tf.AUTO_REUSE):
#       x = tf.get_variable("x",shape=[num_particles, 2],dtype=np.float32,initializer=tf.random_uniform_initializer(-3, 3))
#       w = tf.get_variable("w",dtype=np.float32,initializer=problems.indentity_init(1, 2, 0.01/2),trainable=False)
#       y = tf.get_variable("y",shape=[num_particles, 2],dtype=np.float32,initializer=tf.random_normal_initializer(stddev=0.01/2),trainable=False)
#       wcos = tf.get_variable("wcos",shape=[num_particles, 2],dtype=np.float32,initializer=tf.random_normal_initializer(mean=1.0, stddev=0.01/2),trainable=False)
      
#       sess.run(tf.global_variables_initializer())
#       problem = problems.square_cos(batch_size=num_particles, num_dims=2, mode='test')()

#     # best_position = pso(sess,fitness_fn, max_iter, num_particles, num_dims, -3, 3)
#     # init_pos = [[i-1,i+1] for i in range(num_particles)]
#     # init_vel = [[0,0] for i in range(num_particles)]
#     # best_position, swarm = pso(sess, fitness_fn,problem,x ,max_iter, num_particles, num_dims, -3, 3,init_pos,init_vel)

#     opt = pso(sess,problem,x,fitness_fn=fitness_fn,pop_size=10, dim=2, n_iter=128)
#     sess.run(opt.step())
#     # print(out)
#     # print(sess.run(opt.fit_history))


    # print(opt.g)
    # print(opt.f_p)
    # print(opt.fit_history)

    # print(best_position)
    # print([part.position for part in swarm])