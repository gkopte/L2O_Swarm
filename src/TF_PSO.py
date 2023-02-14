from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import problems

# Construção do grafo
def build_PSO_graph(fitness, better_particle, gbest, x, i):
    # Definição dos hyperparâmetros
    swarm_size = 10
    num_dims = 2
    stddev = 0.01
    dtype = tf.float32
    mode = 'train'
    
    i = tf.constant(0)
    # Inicialização das partículas
    x = tf.get_variable(
        "x",
        shape=[swarm_size, num_dims],
        dtype=dtype,
        initializer=tf.random_uniform_initializer(-3, 3))
        
    # Construção da função de fitness
    # fitness = problems.square_cos(batch_size, num_dims, stddev, dtype, mode)()
    
    # Definição dos placeholders e da operação de otimização
    better_particle = tf.get_variable(
        "best_particle",
        shape=[num_dims],
        dtype=dtype,
        initializer=tf.zeros_initializer(),
        trainable=True)

    gbest = tf.get_variable(
        "global_best_particle",
        shape=[swarm_size, num_dims],
        dtype=dtype,
        initializer=tf.zeros_initializer(),
        trainable=True)
        
    global_best_fitness  = tf.get_variable(
        "global_best_fitness",
        shape=[swarm_size],
        dtype=dtype,
        initializer=tf.zeros_initializer(),
        trainable=True)
        
    pbest = tf.get_variable(
        "personal_best_particle",
        shape=[swarm_size, num_dims],
        dtype=dtype,
        initializer=tf.zeros_initializer(),
        trainable=True)
        
    personal_best_fitness = tf.get_variable(
        "personal_best_fitness",
        shape=[swarm_size],
        dtype=dtype,
        initializer=tf.zeros_initializer(),
        trainable=True)
        
    v = tf.get_variable(
        "particle_velocity",
        shape=[swarm_size, num_dims],
        dtype=dtype,
        initializer=tf.zeros_initializer(),
        trainable=True)
        
    # Definição da atualização das partículas
    with tf.variable_scope("update_particles"):
        random1 = tf.random_uniform(
            shape=[swarm_size, num_dims],
            minval=0,
            maxval=1,
            dtype=dtype)
            
        random2 = tf.random_uniform(
            shape=[swarm_size, num_dims],
            minval=0,
            maxval=1,
            dtype=dtype)

    # Definição dos coeficientes de atualização    
    num_particles = swarm_size # número de partículas
    num_dims = 2 # número de dimensões do espaço de busca
    c1 = 2.05 # constante de aceleração pessoal
    c2 = 2.05 # constante de aceleração social
    w = 0.7298 # inércia

    v = w * v + c1 * random1 * (pbest - x) + c2 * random2 * (gbest - x)

    # Atualização da posição
    x = x + v

    # Atualização da melhor posição pessoal
    better_particle = tf.where(fitness > fitness[tf.argmax(fitness)], x, pbest)

    global_best = tf.reduce_min(fitness)
    gbest = tf.where(fitness == global_best, x, gbest)
    i = i + 1

    return better_particle, gbest, x, i

    # def loop_body(i, count):
    #     return i + 1, count + 1

fitness = problems.square_cos(10, 2, 0.01, tf.float32, 'train')()

with tf.variable_scope("square_cos", reuse=tf.AUTO_REUSE):
    i = tf.constant(0)
    better_particle = tf.get_variable(
        "best_particle",
        shape=[2],
        dtype=tf.float32,
        initializer=tf.zeros_initializer(),
        trainable=True)

    gbest = tf.get_variable(
        "global_best_particle",
        shape=[10, 2],
        dtype=tf.float32,
        initializer=tf.zeros_initializer(),
        trainable=True)

    x = tf.get_variable(
        "x",
        shape=[10, 2],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-3, 3))
    n = 30
    result = tf.while_loop(lambda fitness, better_particle, gbest, x, i: i < n, build_PSO_graph, [fitness, better_particle, gbest, x, i])


    with tf.Session() as sess:
        out = sess.run(result)
        print(out[4])





#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())

#         for i in range(10):
#             new_pos,best_pos, global_besties = sess.run([x,gbest,global_best])
#             fit = sess.run(fitness, feed_dict={x:new_pos})
            
#             # Print the current iteration and best fitness
#             print("Iteration: {} new pos: {}".format(i, new_pos))
#             print("Global best {}".format(fit))

            
#         print("Best position: {}".format(best_pos))

# with tf.variable_scope("square_cos", reuse=tf.AUTO_REUSE):
#     build_PSO_graph()




# import tensorflow as tf

# def loop_body(i, count):
#     return i + 1, count + 1

# def counter(n):
#     i = tf.constant(0)
#     count = tf.constant(0)
#     result = tf.while_loop(lambda i, count: i < n, loop_body, [i, count])
#     return result[1]

# result = counter(5)
# with tf.Session() as sess:
#     print(sess.run(result))
