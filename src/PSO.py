from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# python implementation of particle swarm optimization (PSO)
# minimizing rastrigin and sphere function
 
import random
import math    # cos() for Rastrigin
import copy    # array-copying convenience
import sys     # max float
import problems
from tensorflow.contrib.learn.python.learn import monitored_session as ms
import numpy as np
import tensorflow as tf
import meta
import util

flags = tf.flags
logging = tf.logging

FLAGS = flags.FLAGS
#flags.DEFINE_string("problem", "simple", "Type of problem.")
 
 
# -------fitness functions---------
 
# rastrigin function
def fitness_rastrigin(position):
  fitnessVal = 0.0
  for i in range(len(position)):
    xi = position[i]
    fitnessVal += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
  return fitnessVal
 
#sphere function
def fitness_sphere(position):
    fitnessVal = 0.0
    for i in range(len(position)):
        xi = position[i]
        xi = xi + 0
        fitnessVal += (xi*xi);
    return fitnessVal;
#-------------------------
 
#particle class
class Particle:
  def __init__(self, sess, fitness, problem, x,dim, minx, maxx, seed, pos=None,vel=None):
    self.rnd = random.Random(seed)
 
    # initialize position of the particle with 0.0 value
    self.position = [0.0 for i in range(dim)]
 
     # initialize velocity of the particle with 0.0 value
    self.velocity = [0.0 for i in range(dim)]
 
    # initialize best particle position of the particle with 0.0 value
    self.best_part_pos = [0.0 for i in range(dim)]
 
    # loop dim times to calculate random position and velocity
    # range of position and velocity is [minx, max]
    for i in range(dim):
      if pos is None:
        self.position[i] = ((maxx - minx) *
          self.rnd.random() + minx)
      else:
        self.position[i] = pos[i]

      if vel is None:  
        self.velocity[i] = ((maxx - minx) *
          self.rnd.random() + minx)
      else:
        self.velocity[i] = vel[i]
 
    # compute fitness of particle
    # self.fitness = fitness(sess,self.position) # curr fitness
    print("Inital position: {}".format(self.position))
    #print(type(self.position))
    self.fitness = fitness(sess,problem,x,self.position) # curr fitness
    print("Initial fitness: {}".format(self.fitness))
 
    # initialize best position and fitness of this particle
    self.best_part_pos = copy.copy(self.position)
    self.best_part_fitnessVal = self.fitness # best fitness

 
# particle swarm optimization function
def pso(sess,fitness,problem, x, max_iter, n, dim, minx, maxx,pos=None,vel=None):
  # hyper parameters
  w = 0.729    # inertia
  c1 = 1.49445 # cognitive (particle)
  c2 = 1.49445 # social (swarm)
 
  rnd = random.Random(0)
  
  # create n random particles
  # swarm = [Particle(sess ,fitness, dim, minx, maxx, i, pos[i], vel[i]) for i in range(n)]
  swarm = list()
  #print("pos len: {}".format(len(pos)))

  for i in range(n):
    if pos and vel:
      swarm.append(Particle(sess ,fitness, problem,x,dim, minx, maxx, i, pos[i], vel[i]))
    else:
      swarm.append(Particle(sess ,fitness,problem, x, dim, minx, maxx, i))

  
  # compute the value of best_position and best_fitness in swarm
  best_swarm_pos = [0.0 for i in range(dim)]
  best_swarm_fitnessVal = sys.float_info.max # swarm best
 
  # computer best particle of swarm and it's fitness
  for i in range(n): # check each particle
    if pos:
      swarm[i].position = pos[i]
      swarm[i].fitness = fitness(sess,problem,x,pos[i])
    if vel:
      swarm[i].velocity = vel[i]
    #print(best_swarm_fitnessVal)
    if swarm[i].fitness <best_swarm_fitnessVal:
      best_swarm_fitnessVal = swarm[i].fitness
      best_swarm_pos = copy.copy(swarm[i].position)
 
  # main loop of pso
  Iter = 0
  while (Iter < max_iter) and (best_swarm_fitnessVal > 0.0001):
     
    # after every 10 iterations
    # print iteration number and best fitness value so far
    if Iter % 10 == 0 and Iter> 1:
      print("Iter ="+ str(Iter) +"best fitness = %.3f"% best_swarm_fitnessVal)
 
    for i in range(n): # process each particle
       
      # compute new velocity of curr particle
      for k in range(dim):
        r1 = rnd.random()    # randomizations
        r2 = rnd.random()
     
        swarm[i].velocity[k] = (
                                 (w * swarm[i].velocity[k]) +
                                 (c1 * r1 * (swarm[i].best_part_pos[k] - swarm[i].position[k])) + 
                                 (c2 * r2 * (best_swarm_pos[k] -swarm[i].position[k]))
                              ) 
 
 
        # if velocity[k] is not in [minx, max]
        # then clip it
        if swarm[i].velocity[k] <minx:
          swarm[i].velocity[k] = minx
        elif swarm[i].velocity[k]> maxx:
          swarm[i].velocity[k] = maxx
 
 
      # compute new position using new velocity
      for k in range(dim):
        swarm[i].position[k] += swarm[i].velocity[k]
   
      # compute fitness of new position
      #print(swarm[i].fitness)
      swarm[i].fitness = fitness(sess,problem, x,swarm[i].position)
 
      # is new position a new best for the particle?
      if swarm[i].fitness <swarm[i].best_part_fitnessVal:
        swarm[i].best_part_fitnessVal = swarm[i].fitness
        swarm[i].best_part_pos = copy.copy(swarm[i].position)
 
      # is new position a new best overall?
      if swarm[i].fitness <best_swarm_fitnessVal:
        best_swarm_fitnessVal = swarm[i].fitness
        best_swarm_pos = copy.copy(swarm[i].position)
     
    # for-each particle
    Iter += 1
  #end_while
  return best_swarm_pos, swarm
# end pso

num_dims = 2
num_particles = 10
max_iter = 100
tolerance = 1e-5

def fitness_fn(sess, problem, x,x_val):
  #print("x shape: {}".format(x.shape))
  x_val = np.array(x_val,dtype=np.float32).reshape(1,-1)
  #print("x_val shape: {}".format(x_val.shape))
  sess.run(x.assign(x_val))
  #print(sess.run(x))
  result = sess.run(problem)
  return result.reshape(1, -1).astype(np.float32)

if __name__ == '__main__':
  with tf.Session() as sess:
    
    with tf.variable_scope("square_cos", reuse=tf.AUTO_REUSE):
      x = tf.get_variable("x",shape=[1, 2],dtype=np.float32,initializer=tf.random_uniform_initializer(-3, 3))
      w = tf.get_variable("w",dtype=np.float32,initializer=problems.indentity_init(1, 2, 0.01/2),trainable=False)
      y = tf.get_variable("y",shape=[1, 2],dtype=np.float32,initializer=tf.random_normal_initializer(stddev=0.01/2),trainable=False)
      wcos = tf.get_variable("wcos",shape=[1, 2],dtype=np.float32,initializer=tf.random_normal_initializer(mean=1.0, stddev=0.01/2),trainable=False)
      
      sess.run(tf.global_variables_initializer())
      problem = problems.square_cos(batch_size=1, num_dims=2, mode='train')()
    
    # x_val = np.array([0, 0], dtype=np.float32).reshape(1, -1)
    # print(fitness_fn(sess, problem, x_val))
    
    # x_val = np.array([10, 10], dtype=np.float32).reshape(1, -1)
    # print(fitness_fn(sess, problem, x_val))
    
    # x_val = np.array([0, 0], dtype=np.float32).reshape(1, -1)
    # print(fitness_fn(sess, problem, x_val))

    # best_position = pso(sess,fitness_fn, max_iter, num_particles, num_dims, -3, 3)
    init_pos = [[i-1,i+1] for i in range(num_particles)]
    init_vel = [[0,0] for i in range(num_particles)]
    best_position, swarm = pso(sess, fitness_fn,problem, x,max_iter, num_particles, num_dims, -3, 3,init_pos,init_vel)

  print(best_position)
  print([part.position for part in swarm])
    
    # print(fitness_fn(problem,[1,2],batch_size=num_particles,num_dims=2))
    # Define the PSO parameter  
    # print(fitness_fn(sess,[1,2]))
        # return result.reshape(-1,1).tolist()





  # Run the PSO optimization
  # def pso(fitness, max_iter, n, dim, minx, maxx):
  # best_position = pso(fitness, max_iter, num_particles, dim, -10.0, 10.0)


 
#----------------------------
# Driver code for rastrigin function
 
# print("\nBegin particle swarm optimization on rastrigin function\n")
# dim = 3
# # def square_cos(batch_size=128, num_dims=None,  stddev=0.01, dtype=tf.float32, mode='train'):
# # problem = problems.square_cos(batch_size=128, num_dims=dim,  stddev=0.01, dtype=tf.float32, mode='train')
# problem, net_config, net_assignments = util.get_config(FLAGS.problem)
# fitness = problem()
# print(fitness)
# # sub_x, sub_constants = meta._get_variables(problem)
# writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

# with ms.MonitoredSession() as sess:
#     tf.get_default_graph().finalize()
#     fitness_batch_numpy = sess.run(fitness)


# fitness_batch = fitness_batch_numpy.tolist()
# print(fitness_batch) 
# print(len(fitness_batch))
 
# print("Goal is to minimize Rastrigin's function in"+ str(dim) +"variables")
# print("Function has known min = 0.0 at (", end="")
# for i in range(dim-1):
#   print("0,", end="")
# print("0)")
 
# num_particles = 50
# max_iter = 100
 
# print("Setting num_particles ="+ str(num_particles))
# print("Setting max_iter    ="+ str(max_iter))
# print("\nStarting PSO algorithm\n")
 
 
 
# best_position = pso(fitness, max_iter, num_particles, dim, -10.0, 10.0)
 
# print("\nPSO completed\n")
# print("\nBest solution found:")
# print(["%.6f"%best_position[k] for k in range(dim)])
# fitnessVal = fitness(best_position)
# print("fitness of best solution = %.6f"% fitnessVal)
 
# print("\nEnd particle swarm for rastrigin function\n")
