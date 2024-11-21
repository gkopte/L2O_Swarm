

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
from six.moves import xrange
import tensorflow as tf
# tf.enable_eager_execution()
import numpy as np

import meta
import util
import pdb

from tensorflow.contrib.learn.python.learn import monitored_session as ms
from tensorflow.python import debug as tf_debug
import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

experiment_name = "IECON2023_train"

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
debug_mode = False


FLAGS = flags.FLAGS
flags.DEFINE_string("pre_trained_path", None, "Path for pre trained meta-optimizer.")
flags.DEFINE_string("save_path", None, "Path for saved meta-optimizer.")
flags.DEFINE_integer("num_epochs", 1000, "Number of training epochs.")
flags.DEFINE_integer("log_period", 100, "Log period.")
flags.DEFINE_integer("evaluation_period", 100, "Evaluation period.")
flags.DEFINE_integer("evaluation_epochs", 20, "Number of evaluation epochs.")
flags.DEFINE_integer("num_particle", 4, "Number of evaluation epochs.") # The number of particles
flags.DEFINE_string("problem", "simple", "Type of problem.")
flags.DEFINE_integer("num_steps", 100,
                     "Number of optimization steps per epoch.")
flags.DEFINE_integer("unroll_length", 20, "Meta-optimizer unroll length.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_boolean("second_derivatives", False, "Use second derivatives.")
flags.DEFINE_string("im_loss_option", "mse", "function used in the imitation learning loss")

mlflow.set_tag("mlflow.runName", FLAGS.save_path)   
mlflow.log_param("pre_trained_path", FLAGS.pre_trained_path)
mlflow.log_param("save_path", FLAGS.save_path)
mlflow.log_param("num_epochs", FLAGS.num_epochs)
mlflow.log_param("log_period", FLAGS.log_period)
mlflow.log_param("evaluation_period", FLAGS.evaluation_period)
mlflow.log_param("evaluation_epochs", FLAGS.evaluation_epochs)
mlflow.log_param("num_particle", FLAGS.num_particle)
mlflow.log_param("problem", FLAGS.problem)
mlflow.log_param("num_steps", FLAGS.num_steps)
mlflow.log_param("unroll_length", FLAGS.unroll_length)
mlflow.log_param("learning_rate", FLAGS.learning_rate)
mlflow.log_param("second_derivatives", FLAGS.second_derivatives)
mlflow.log_param("im_loss_option", FLAGS.im_loss_option)



def main(_):

  # Configuration.
  num_unrolls = FLAGS.num_steps // FLAGS.unroll_length

  if FLAGS.pre_trained_path is None:
    problem, net_config, net_assignments = util.get_config(FLAGS.problem)
  else:
    print('PRE TRAINED MODEL PATH: ', FLAGS.pre_trained_path)
    problem, net_config, net_assignments = util.get_config(FLAGS.problem,
                                                        FLAGS.pre_trained_path)
    
  optimizer = meta.MetaOptimizer(FLAGS.problem, FLAGS.num_particle ,**net_config)
  if FLAGS.save_path is not None:
    if not os.path.exists(FLAGS.save_path):
      os.mkdir(FLAGS.save_path)
      path = None
#      raise ValueError("Folder {} already exists".format(FLAGS.save_path))
    else:
      if os.path.exists('{}/loss-record.pickle'.format(FLAGS.save_path)):
        path = FLAGS.save_path
      else:
        path = None
  # Problem.
  

  # Optimizer setup.
  
  minimize = optimizer.meta_minimize(
              problem, FLAGS.unroll_length,
              FLAGS.im_loss_option,
              learning_rate=FLAGS.learning_rate,
              net_assignments=net_assignments,
              model_path = path,
              second_derivatives=FLAGS.second_derivatives)

  step, update, reset, cost_op, x_final, test, fc_weights, fc_bias, fc_va= minimize
#  saver=tf.train.Saver()
  # Creating a summary writer
  writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
  #hook = tf_debug.TensorBoardDebugHook("DESKTOP-TDLPICQ:6000")
  with ms.MonitoredSession() as sess:
    # Prevent accidental changes to the graph.
    tf.get_default_graph().finalize()
#    Step=[step for i in range(len(cost_op))]
    best_evaluation = float("inf")
    total_time = 0
    total_cost = 0
    loss_record = []
    constants = []
    if debug_mode:
      pdb.set_trace()
    for e in xrange(FLAGS.num_epochs):
      # Training.
      time, cost, constant, Weights = util.run_epoch(sess, cost_op, [update, step], reset,
                                  num_unrolls, test, [fc_weights, fc_bias, fc_va])
      cost= sum(cost)/len(cost_op)
      total_time += time
      total_cost += cost
      loss_record.append(cost)
      constants.append(constant)
      # Logging.
      if (e + 1) % FLAGS.log_period == 0:
        util.print_stats("Epoch {}".format(e + 1), total_cost, total_time,
                         FLAGS.log_period)
        total_time = 0
        total_cost = 0

      # Evaluation.
      if (e + 1) % FLAGS.evaluation_period == 0:
        eval_cost = 0
        eval_time = 0
        for _ in xrange(FLAGS.evaluation_epochs):
          time, cost, constant, weights = util.run_epoch(sess, cost_op, [update, step], reset,
                                  num_unrolls, test, [fc_weights, fc_bias, fc_va])
#          cost/=len(cost_op)
          eval_time += time
          eval_cost += sum(cost)/len(cost_op)

        util.print_stats("EVALUATION", eval_cost, eval_time,
                         FLAGS.evaluation_epochs)

        if FLAGS.save_path is not None and eval_cost < best_evaluation:
          print("Removing previously saved meta-optimizer")
          for f in os.listdir(FLAGS.save_path):
            os.remove(os.path.join(FLAGS.save_path, f))
          print("Saving meta-optimizer to {}".format(FLAGS.save_path))
#          saver.save(sess,'./quadratic/quadratic.ckpt',global_step = e)
          optimizer.save(sess, FLAGS.save_path)
          with open(FLAGS.save_path+'/loss_record.pickle','wb') as l_record:
            record = {'loss_record':loss_record, 'fc_weights':sess.run(weights[0]), \
                'fc_bias':sess.run(weights[1]), 'fc_va':sess.run(weights[2]), 'constant':sess.run(constant)}
            pickle.dump(record, l_record)
          best_evaluation = eval_cost
#    fc_weights = np.array(sess.run(fc_weights))

    try:
      mlflow.log_artifact(FLAGS.path+'_log.txt')
    except:
      print("Training log not found")
    try:
      mlflow.log_artifact(FLAGS.path+'/cw.l2l')
    except:
      print("Model not found")
    try:
      mlflow.log_artifact(FLAGS.path+'/loss_record.pickle')
    except:
      print("Loss record not found")
    try:
      mlflow.log_artifact(FLAGS.path+'/evaluate_record.pickle')
    except:
      print("Evaluate record not found")
      
    mlflow.end_run()

if __name__ == "__main__":
  tf.app.run()
