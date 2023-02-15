import tensorflow as tf
tf.enable_eager_execution()
# import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


np.random.seed(123)
tf.random.set_random_seed(123)
# tf.random.set_random_seed(123)

class pso:
    def __init__(
        self,
        fitness_fn,
        pop_size=100,
        dim=2,
        n_iter=200,
        b=0.9,
        c1=0.8,
        c2=0.5,
        x_min=-1,
        x_max=1,
        ):
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
        self.f_p = self.fitness_fn(self.x)
        self.fit_history = []
        self.g = self.p[tf.math.argmin(input=self.f_p).numpy()[0]]
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
        f_x = self.fitness_fn(self.x)
        self.fit_history.append(tf.reduce_mean(f_x).numpy())
        self.p = tf.where(f_x < self.f_p, self.x, self.p)
        self.f_p = tf.where(f_x < self.f_p, f_x, self.f_p)

    def update_g_best(self):
        """Update the *g-best* position. 
        """
        self.g = self.p[tf.math.argmin(input=self.f_p).numpy()[0]]

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

def objective_function(X):
    return tf.math.sqrt(X[:,0]**2 + X[:,1]**2)[:,None]

def fitness_function():
    def f(X):
        return objective_function(X)
    return f

opt = pso(fitness_fn=fitness_function(), n_iter=1)

print(opt.x)
## Define the grid for future plotting:
xlist = np.linspace(-1.0, 1.0, 100)
ylist = np.linspace(-1.0, 1.0, 100)
X, Y = np.meshgrid(xlist, ylist)
Z = - np.sqrt(X**2 + Y**2)


def snapshot(i):
    opt.train()
    plt.contourf(X, Y, Z, cmap='rainbow', levels=50)
    plt.scatter(opt.x[:,0],opt.x[:,1], color='b')

fig = plt.figure(figsize=(8, 8), dpi=100)

# anim = animation.FuncAnimation(fig,snapshot,frames=60)
# anim.save("PSO_tensorflow.mp4", fps=6)