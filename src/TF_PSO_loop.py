from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import problems

# Construção do grafo
def build_PSO_graph():
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
        
    # Construção da função de fitness
    fitness = problems.square_cos(swarm_size, num_dims, stddev, dtype, mode)()
    
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

    # Definição dos coeficientes de atualização    
    num_particles = swarm_size # número de partículas
    num_dims = 2 # número de dimensões do espaço de busca
    c1 = 2.05 # constante de aceleração pessoal
    c2 = 2.05 # constante de aceleração social
    w = 0.7298 # inércia

    # Loop para atualização das partículas
    i = tf.constant(0)
    def loop_cond(i, *args):
        return tf.less(i, 128)

    def loop_body(i, x, v, fitness, better_particle, pbest, gbest):
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

        # Atualização da velocidade
        v = w * v + c1 * random1 * (pbest - x) + c2 * random2 * (gbest - x)

        # Atualização da posição
        x = x + v

        # Atualização da melhor posição pessoal
        better_particle = tf.where(fitness > fitness[tf.argmax(fitness)], x, pbest)

        # Atualização da melhor posição pessoal

        better_particle = tf.where(fitness > fitness[tf.argmax(fitness)], x, pbest)
        pbest = tf.where(fitness > personal_best_fitness, x, pbest)
        personal_best_fitness = tf.where(fitness > personal_best_fitness, fitness, personal_best_fitness)
        # Atualização da melhor posição global

        gbest = tf.where(global_best_fitness > fitness, gbest, x)
        global_best_fitness = tf.where(global_best_fitness > fitness, global_best_fitness, fitness)
        # Definição da operação de otimização

        update_particles_op = tf.group(*[x.assign(better_particle),
        pbest.assign(pbest),
        personal_best_fitness.assign(personal_best_fitness),
        gbest.assign(gbest),
        global_best_fitness.assign(global_best_fitness),
        v.assign(v)])
        # Inicialização das variáveis

        init_op = tf.global_variables_initializer()
        # Criação da sessão e execução do grafo

        with tf.Session() as sess:
            sess.run(init_op)
            for i in range(128):
                sess.run(update_particles_op)
                if i % 10 == 0:
                    fitness_val, x_val, pbest_val, gbest_val = sess.run([fitness, x, pbest, gbest])
                print("Iteration:", i)
                print("Fitness:", fitness_val)
                print("Position:", x_val)
                print("Personal Best:", pbest_val)
                print("Global Best:", gbest_val)
