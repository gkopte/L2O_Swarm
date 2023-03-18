from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import problems
import pdb

debug_mode = True

# Construção do grafo
      
def build_PSO_graph(x_ini,v_ini):
    # Definição dos hyperparâmetros
    swarm_size = 10
    num_dims = 2
    stddev = 0.01
    dtype = tf.float32
    mode = 'test'
    
    # Inicialização das partículas
    x = tf.get_variable(
        "x",
        shape=[swarm_size, num_dims],
        dtype=dtype,
        initializer=tf.random_uniform_initializer(-3, 3),
        trainable=True)
    x.assign(x_ini)
        
    # Construção da função de fitness
    fitness = problems.square_cos(swarm_size, num_dims, stddev, dtype, mode)()
    
    # Definição dos placeholders e da operação de otimização
    better_particle = tf.get_variable(
        "best_particle",
        shape=[num_dims],
        dtype=dtype,
        initializer=tf.constant_initializer(100),
        trainable=True)

    gbest = tf.get_variable(
        "global_best_particle",
        shape=[swarm_size, num_dims],
        dtype=dtype,
        initializer=tf.constant_initializer(100),
        trainable=True)
    # gbest.assign(gbest_init)
        
    # global_best_fitness  = tf.get_variable(
    #     "global_best_fitness",
    #     shape=[swarm_size],
    #     dtype=dtype,
    #     initializer=tf.zeros_initializer(),
    #     trainable=True)
        
    pbest = tf.get_variable(
        "personal_best_particle",
        shape=[swarm_size, num_dims],
        dtype=dtype,
        initializer=tf.constant_initializer(100.0),
        trainable=True)
    # pbest.assign(pbest_init)
    if debug_mode is True:
       pdb.set_trace()
        
    # personal_best_fitness = tf.get_variable(
    #     "personal_best_fitness",
    #     shape=[swarm_size],
    #     dtype=dtype,
    #     initializer=tf.zeros_initializer(),
    #     trainable=True)
        
    v = tf.get_variable(
        "particle_velocity",
        shape=[swarm_size, num_dims],
        dtype=dtype,
        initializer=tf.zeros_initializer(),
        trainable=True)
    v.assign(v_ini)
        
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
    if debug_mode is True:
       pdb.set_trace()

    for i in range(10):

        v = w * v + c1 * random1 * (pbest - x) + c2 * random2 * (gbest - x)

        # Atualização da posição
        x = x + v

        if debug_mode is True:
            pdb.set_trace()
        # Atualização da melhor posição pessoal
        better_particle = tf.where(fitness > fitness[tf.argmin(fitness)], x, pbest)
        

        global_best = tf.reduce_min(fitness)
        gbest = tf.where(fitness == global_best, x, gbest)

        out = sess.run((global_best,pbest, gbest,x,v, better_particle ,fitness))
        print(out)
    return out

    # print('IN global_best ',out[0])
    # print('IN gbest ',out[1])
    # print('IN better_particle ',out[2])
    # print('IN fitness',out[3])
    return out



if __name__ == '__main__':
  with tf.Session() as sess:
    
    with tf.variable_scope("square_cos", reuse=tf.AUTO_REUSE):
        x = tf.get_variable("x",shape=[10, 2],dtype=np.float32,initializer=tf.random_uniform_initializer(-3, 3))
        v = tf.get_variable("x",shape=[10, 2],dtype=np.float32,initializer=tf.random_uniform_initializer(-3, 3))
        w = tf.get_variable("w",dtype=np.float32,initializer=problems.indentity_init(1, 2, 0.01/2),trainable=False)
        y = tf.get_variable("y",shape=[10, 2],dtype=np.float32,initializer=tf.random_normal_initializer(stddev=0.01/2),trainable=False)
        wcos = tf.get_variable("wcos",shape=[10, 2],dtype=np.float32,initializer=tf.random_normal_initializer(mean=1.0, stddev=0.01/2),trainable=False)
        gbest = tf.get_variable(
        "global_best_particle",
        shape=[10, 2],
        dtype=np.float32,
        initializer=tf.zeros_initializer(),
        trainable=True)

        pbest = tf.get_variable(
        "personal_best_particle",
        shape=[10, 2],
        dtype=np.float32,
        initializer=tf.zeros_initializer(),
        trainable=True)

        v = tf.get_variable(
        "particle_velocity",
        shape=[10, 2],
        dtype=np.float32,
        initializer=tf.zeros_initializer(),
        trainable=True)
        
        problem = problems.square_cos(batch_size=10, num_dims=2, mode='test')()
        sess.run(tf.global_variables_initializer())
        output = []
        # for i in range(10):
            # global_best, gbest, better_particle ,fitness = sess.run(build_PSO_graph(x,v))
        global_best, pbest, gbest,x,v, better_particle ,fitness = build_PSO_graph(x,v)

        print('best_particle ',gbest)
        print('x ',better_particle)
        print('fitness ',fitness)





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
