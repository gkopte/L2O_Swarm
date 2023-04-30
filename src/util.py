
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from timeit import default_timer as timer

import numpy as np
from six.moves import xrange

import problems
import pdb
import tensorflow as tf

debug_mode = False

def run_epoch(sess, cost_op, ops, reset, num_unrolls, var1, var2):
  """Runs one optimization epoch."""
  start = timer()
  sess.run(reset)
  for j in xrange(num_unrolls):
    cost=[]
    step = sess.run(ops)
    for i in range(len(cost_op)):
      sub_cost = (sess.run([cost_op[i]]) + step)[0] 
      cost.append(sub_cost)
    print('cost', cost)
  print ('done one epoch')
  return timer() - start, cost, var1, var2

def eval_run_epoch(sess, cost_op, ops, reset, num_unrolls, var1, var2):
  """Runs one optimization epoch."""
  start = timer()
  sess.run(reset)
  cost_all =[]
  x_history = []
  y_history = []
  for _ in xrange(num_unrolls):
    step = sess.run(ops)
    cost=[]
    self_loss = []
    step = []
    step.append(step)
    # ops = tf.get_default_graph()
    x_ = sess.run(var1[0])
    # y_ = sess.run(var1[1])
    # print(x_[:,0,:])
    # print(x_[:,-1,:])
    # print(y_[:,0,:])
    # print(y_[:,-1,:])
    x_history.append([x_[:,0,:],x_[:,-1,:]])
    # y_history.append(y_[:,0,:],y_[:,-1,:])
    
    for i in range(len(cost_op)):
      sub_self_loss  = sess.run([cost_op[i]])
      sub_cost = (sub_self_loss + step)[0] 
      # sub_cost = (sess.run([cost_op[i]]) + step)[0] 
      self_loss.append(sub_self_loss)
      cost.append(sub_cost)
      

    if debug_mode and (_ == num_unrolls-1):
      pdb.set_trace()
    # print(f'\nsub_cost:{np.mean(cost)} = sub_self_loss({np.mean(self_loss)}) + loss({np.mean(cost)-np.mean(self_loss)})')
    # print('\var1 his', var1_history)
    # print('cost', cost)
    # print('mean cost', np.mean(cost))
    # print('min cost', np.min(cost))
    cost_all.append(cost)
  return timer() - start, cost_all,  var2, x_history #, y_history


def print_stats(header, total_error, total_time, n):
  """Prints experiment statistics."""
  print(header)
  if total_error!=0:
    # print(n)
    # print(total_error)
    print("Log Mean Final Error: {:.2f}".format(np.log10(total_error / n)))
    print("Mean epoch time: {:.2f} s".format(total_time / n))  
  else:
    print("Log Mean Final Error: inf")
    print("Mean epoch time: {:.2f} s".format(total_time / n))

def get_net_path(name, path):
  return None if path is None else os.path.join(path, name + ".l2l")


def get_default_net_config(name, path):
  return {
      "net": "CoordinateWiseDeepLSTM",
      "net_options": {
          "layers": (20, 20),
          "preprocess_name": "LogAndSign",
          "preprocess_options": {"k": 5},
          "scale": 0.01,
      },
      "net_path": get_net_path(name, path)
  }


def get_config(problem_name, path=None, mode='train'):
  """Returns problem configuration."""
  if problem_name == "simple":
    problem = problems.simple()
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (), "initializer": "zeros"},
        "net_path": get_net_path("cw", path)
    }}
    net_assignments = None
  elif problem_name == "simple-multi":
    problem = problems.simple_multi_optimizer()
    net_config = {
        "cw": {
            "net": "CoordinateWiseDeepLSTM",
            "net_options": {"layers": (), "initializer": "zeros"},
            "net_path": get_net_path("cw", path)
        },
        "adam": {
            "net": "Adam",
            "net_options": {"learning_rate": 0.1}
        }
    }
    net_assignments = [("cw", ["x_0"]), ("adam", ["x_1"])]
  elif problem_name == "quadratic":
    problem = problems.quadratic(batch_size=128, num_dims=5)
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (20, 20)},
        "net_path": get_net_path("cw", path)
    }}
    net_assignments = None
  elif problem_name == "mnist":
    mode = "train" if path is None else "test"
    problem = problems.mnist(layers=(20,), mode=mode)
    net_config = {"cw": get_default_net_config("cw", path)}
    net_assignments = None
  elif problem_name == "cifar":
    mode = "train" if path is None else "test"
    problem = problems.cifar10("cifar10",
                               conv_channels=(16, 16, 16),
                               linear_layers=(32,),
                               mode=mode)
    net_config = {"cw": get_default_net_config("cw", path)}
    net_assignments = None
  elif problem_name == "cifar-multi":
    mode = "train" if path is None else "test"
    problem = problems.cifar10("cifar10",
                               conv_channels=(16, 16, 16),
                               linear_layers=(32,),
                               mode=mode)
    net_config = {
        "conv": get_default_net_config("conv", path),
        "fc": get_default_net_config("fc", path)
    }
    conv_vars = ["conv_net_2d/conv_2d_{}/w".format(i) for i in xrange(3)]
    fc_vars = ["conv_net_2d/conv_2d_{}/b".format(i) for i in xrange(3)]
    fc_vars += ["conv_net_2d/batch_norm_{}/beta".format(i) for i in xrange(3)]
    fc_vars += ["mlp/linear_{}/w".format(i) for i in xrange(2)]
    fc_vars += ["mlp/linear_{}/b".format(i) for i in xrange(2)]
    fc_vars += ["mlp/batch_norm/beta"]
    net_assignments = [("conv", conv_vars), ("fc", fc_vars)]
  elif  "square_cos" in problem_name:
    num_dims = int(problem_name.split('_')[-1])
    problem = problems.square_cos(batch_size=128, num_dims=num_dims, mode=mode)
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (20, 20)},
        "net_path": get_net_path("cw", path)
    }}
    net_assignments = None
  elif problem_name == "protein_dock":
    problem = problems.protein_dock(batch_size=125, num_dims=12)
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (20, 20)},
        "net_path": get_net_path("cw", path)
    }}
    net_assignments = None
  else:
    raise ValueError("{} is not a valid problem".format(problem_name))

  return problem, net_config, net_assignments
