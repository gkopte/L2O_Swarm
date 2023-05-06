
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import monitored_session as ms

import math
import meta
import util
import os
import pickle
import pdb
import random
from tensorflow.python import debug as tf_debug
import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

experiment_name = "experiment_test1"

# Get the experiment by name
experiment = client.get_experiment_by_name(experiment_name)

# If the experiment exists and is deleted, restore it
if experiment and experiment.lifecycle_stage == "deleted":
    client.restore_experiment(experiment.experiment_id)

# Set the restored experiment as the active experiment
mlflow.set_experiment(experiment_name)



mlflow.start_run()



flags = tf.flags
logging = tf.logging


FLAGS = flags.FLAGS
flags.DEFINE_string("optimizer", "L2L", "Optimizer.")
flags.DEFINE_string("path", None, "Path to saved meta-optimizer network.")
flags.DEFINE_integer("num_epochs", 1, "Number of evaluation epochs.")
flags.DEFINE_integer("seed", None, "Seed for TensorFlow's RNG.")
# The number of particles
flags.DEFINE_integer("num_particle", 4, "Number of evaluation epochs.")
flags.DEFINE_string("problem", "simple", "Type of problem.")
flags.DEFINE_integer("num_steps", 250,
                     "Number of optimization steps per epoch.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_string("im_loss_option", "",
                    "function used in the imitation learning loss")

mlflow.set_tag("mlflow.runName", str(FLAGS.path))
mlflow.log_param("optimizer", FLAGS.optimizer)
mlflow.log_param("path", FLAGS.path)
mlflow.log_param("num_epochs", FLAGS.num_epochs)
mlflow.log_param("num_particle", FLAGS.num_particle)
mlflow.log_param("problem", FLAGS.problem)
mlflow.log_param("learning_rate", FLAGS.learning_rate)
mlflow.log_param("im_loss_option", FLAGS.im_loss_option)


def main(_):
    # Configuration.
    num_unrolls = FLAGS.num_steps
    seed_value = random.randint(0, 100)
    # if FLAGS.seed:
    tf.set_random_seed(seed_value)

    # Problem.
    problem, net_config, net_assignments = util.get_config(FLAGS.problem,
                                                           FLAGS.path, mode='test')

    # Optimizer setup.
    if FLAGS.optimizer == "Adam":
        cost_op = problem()
        problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        problem_reset = tf.variables_initializer(problem_vars)

        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
        update = optimizer.minimize(cost_op)
        reset = [problem_reset, optimizer_reset]

        # x_final = tf.constant(0, shape=(128, 7, 2), dtype=tf.float32)
        # constant = tf.constant(0, shape=(128, 7, 2), dtype=tf.float32)

    elif FLAGS.optimizer == "L2L":
        if FLAGS.path is None:
            logging.warning("Evaluating untrained L2L optimizer")
        optimizer = meta.MetaOptimizer(
            FLAGS.problem, FLAGS.num_particle,  **net_config)
        meta_loss = optimizer.meta_loss(
            problem, 1, FLAGS.im_loss_option, net_assignments=net_assignments, model_path=FLAGS.path)
        loss, update, reset, cost_op, x_final, constant = meta_loss
    else:
        raise ValueError("{} is not a valid optimizer".format(FLAGS.optimizer))
    # writer = tf.summary.FileWriter('./graphs_eval', tf.get_default_graph())
    # debug_hook = tf_debug.TensorBoardDebugHook("DESKTOP-TDLPICQ:6000")
    with ms.MonitoredSession() as sess:
        # Prevent accidental changes to the graph.
        tf.get_default_graph().finalize()
        
        min_loss_record = []
        all_time_loss_record = []
        total_time = 0
        total_cost = 0
        # x_record = [[sess.run(item) for item in x_final]]
        x_history = []
        # y_history = []
        #pdb.set_trace()
        for epoch in xrange(FLAGS.num_epochs):
            # Training.
            print("Epoch: ",epoch+1)
            # x_init = sess.run(x_final)

            time, cost,  constants, x_ = util.eval_run_epoch(sess, cost_op, [update], reset,
                                                              num_unrolls, x_final, constant)

            total_time += time
            all_time_loss_record.append(cost)
            x_history.append(x_)
            # print('x final: ', x_[-1][1])
            print('cost: ', cost[-1])
            print('min cost: ', min(min(cost)))
            # print(type(x_))
            # x_end = sess.run(x_final)
            # pdb.set_trace()

            mlflow.log_metric("epoch_min_cost", min(min(cost)), step=epoch+1)
            mlflow.log_metric("epoch_max_cost", min(min(cost)), step=epoch+1)

            # pdb.set_trace()
            for step, step_cost in enumerate(cost):
                step_mean = sum(step_cost) / len(step_cost)
                variance = sum((x - step_mean) ** 2 for x in step_cost) / len(step_cost)
                std_dev = math.sqrt(variance)

                mlflow.log_metric("epoch_"+ str(epoch+1) +"_cost_min", min(step_cost), step=step+1)
                mlflow.log_metric("epoch_"+ str(epoch+1) +"_cost_max", max(step_cost), step=step+1)

                mlflow.log_metric("epoch_"+ str(epoch+1) +"_cost_avg", step_mean, step=step+1)
                mlflow.log_metric("epoch_"+ str(epoch+1) +"_cost_std", std_dev, step=step+1)
                mlflow.log_metric("epoch_"+ str(epoch+1) +"_cost_avg_m_std", step_mean - std_dev, step=step+1)
                mlflow.log_metric("epoch_"+ str(epoch+1) +"_cost_avg_p_std", step_mean + std_dev, step=step+1)
                for part, part_cost in enumerate(step_cost):
                    mlflow.log_metric("epoch_"+str(epoch+1)+"_particle_"+ str(part+1)  +"_cost", part_cost, step=step+1)
                



        with open('./{}/evaluate_record.pickle'.format(FLAGS.path), 'wb') as l_record:
            record = {'all_time_loss_record': all_time_loss_record, 'loss': cost,
                      'constants': [sess.run(item) for item in constants],
                      'x':x_history}
            pickle.dump(record, l_record)
        # Results.
        util.print_stats("Epoch {}".format(FLAGS.num_epochs), total_cost,
                         total_time, FLAGS.num_epochs)
        
        # Assuming you are inside an active MLflow run
        mlflow.log_artifact(FLAGS.path+'/cw.l2l')
        mlflow.log_artifact(FLAGS.path+'/loss_record.pickle')
        mlflow.log_artifact(FLAGS.path+'/evaluate_record.pickle')
        mlflow.end_run()
if __name__ == "__main__":
    tf.app.run()
