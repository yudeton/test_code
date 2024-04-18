#!/usr/bin/env python3
# coding: utf-8
import importlib
import sys
import yaml
import rospy

importlib.reload(sys)
import argparse
import csv
import math
import time
from collections import namedtuple
from math import pi
from os import path

import geometry_msgs.msg
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import plotly.graph_objs as go
import plotly.offline as py
import roboticstoolbox as rtb
import sympy as sp
# import dyna_space
from interface_control.msg import (cal_cmd, cal_process, cal_result, dyna_data,
                                dyna_space_data, optimal_design,
                                optimal_random, specified_parameter_design, tested_model_name, arm_structure)
from matplotlib import cm
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from mpl_toolkits.mplot3d import Axes3D
from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.drawing.image import Image
from openpyxl.styles import Alignment, Font, colors
from openpyxl.utils import get_column_letter
from plotly.offline import download_plotlyjs, iplot, plot
from scipy.interpolate import make_interp_spline  # draw smooth
from spatialmath import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from dqn import DQN

np.set_printoptions(
    linewidth=100,
    formatter={"float": lambda x: f"{x:8.4g}" if abs(x) > 1e-10 else f"{0:8.4g}"},
)

from arm_workspace import arm_workspace_plane
# from robot_urdf import RandomRobot
from motor_module import motor_data
from random_robot import RandomRobot
from modular_robot_6dof import modular_robot_6dof
# DRL_optimization api
import sys
import os
import torch.nn as nn
import torch.nn.functional as F
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径

import gym
import torch
import datetime
import numpy as np
from common.utils import save_results, make_dir
from common.utils import plot_rewards
from dqn import DQN

import argparse
import random

import matplotlib.pyplot as plt
# from RobotOptEnv_dynamixel import RobotOptEnv, RobotOptEnv_3dof, RobotOptEnv_5dof
from RobotOptEnv_dynamixel_v2 import RobotOptEnv, RobotOptEnv_3dof, RobotOptEnv_5dof
import tensorboardX
import yaml

# add
import tensorflow as tf
from tf_agents.agents.dqn.dqn_agent import DqnAgent, DdqnAgent
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import categorical_q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.policies import policy_saver # add
from tf_agents.policies import py_tf_eager_policy # add
# add
import io
import os
import shutil
import tempfile
import zipfile
import itertools

files = None
# add

file_path = curr_path + "/outputs/"
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
curr_time_real = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
# curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
# tempdir = curr__ + "/C51_outputs/" + \
#             '/' + curr_time + '/models/'  # 保存模型的路径
# tb = tensorboardX.SummaryWriter()
tb = None
# tensorboard_callback = tensorboardX.(log_dir=log_dir, histogram_freq=1)
algo_name = "dqn"  # 算法名称
env_name = 'dqn_RobotOptEnv'  # 环境名称

class drl_optimization:
    def __init__(self):
        # self.test = 0
        self.robot = modular_robot_6dof()
        self.env = RobotOptEnv()
        # self.config = Config()

    def env_agent_config(self, cfg, algorithm, seed=1):
        ''' 创建环境和智能体
        '''
        # num_iterations = 15000 # @param {type:"integer"}

        # initial_collect_steps = 1000  # @param {type:"integer"}
        # collect_steps_per_iteration = 1  # @param {type:"integer"}
        # replay_buffer_capacity = 100000  # @param {type:"integer"}

        fc_layer_params = (100,)

        # batch_size = 64  # @param {type:"integer"}
        learning_rate = 1e-3  # @param {type:"number"}
        gamma = 0.99
        # log_interval = 200  # @param {type:"integer"}

        num_atoms = 51  # @param {type:"integer"}
        min_q_value = -100  # @param {type:"integer"}
        max_q_value = 50  # @param {type:"integer"}
        n_step_update = 2  # @param {type:"integer"}

        # num_eval_episodes = 10  # @param {type:"integer"}
        # eval_interval = 1000  # @param {type:"integer"}

        train_py_env = suite_gym.wrap_env(self.env)
        # eval_py_env = suite_gym.load(self.env)

        env = tf_py_environment.TFPyEnvironment(train_py_env)
        # eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

        categorical_q_net = categorical_q_network.CategoricalQNetwork(
            env.observation_spec(),
            env.action_spec(),
            num_atoms=num_atoms,
            fc_layer_params=fc_layer_params)

        dqn_network = QNetwork(
            env.observation_spec(),
            env.action_spec(),
            fc_layer_params=fc_layer_params)

        ddqn_network = QNetwork(
            env.observation_spec(),
            env.action_spec(),
            fc_layer_params=fc_layer_params)
        
        ddqn_network = QNetwork(
            env.observation_spec(),
            env.action_spec(),
            fc_layer_params=fc_layer_params)
        

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        self.global_step = tf.compat.v1.train.get_or_create_global_step()

        # train_step_counter = tf.Variable(0)
        # agent = DDQNAgent(self.config)  # 创建智能体
        if algorithm == 'DQN':
            agent = DqnAgent(
                env.time_step_spec(),
                env.action_spec(),
                q_network = dqn_network,
                optimizer = optimizer,
                n_step_update=n_step_update,
                td_errors_loss_fn = common.element_wise_squared_loss,
                gamma=gamma,
                train_step_counter = self.global_step)
            agent.initialize()
            rospy.loginfo("DRL algorithm init: %s", algorithm)
        elif algorithm == 'DDQN':
            agent = DdqnAgent(
                env.time_step_spec(),
                env.action_spec(),
                q_network = ddqn_network,
                optimizer = optimizer,
                n_step_update=n_step_update,
                td_errors_loss_fn = common.element_wise_squared_loss,
                gamma=gamma,
                train_step_counter = self.global_step)
            agent.initialize()
            rospy.loginfo("DRL algorithm init: %s", algorithm)
        elif algorithm == 'C51':
            agent = categorical_dqn_agent.CategoricalDqnAgent(
                env.time_step_spec(),
                env.action_spec(),
                categorical_q_network=categorical_q_net,
                optimizer=optimizer,
                min_q_value=min_q_value,
                max_q_value=max_q_value,
                n_step_update=n_step_update,
                td_errors_loss_fn=common.element_wise_squared_loss,
                gamma=gamma,
                train_step_counter=self.global_step)
            agent.initialize()
            rospy.loginfo("DRL algorithm init: %s", algorithm)

        # dqn_agent = DqnAgent(
        #     env.time_step_spec(),
        #     env.action_spec(),
        #     q_network = dqn_network,
        #     optimizer = optimizer,
        #     td_errors_loss_fn = common.element_wise_squared_loss,
        #     train_step_counter = self.global_step)

        # ddqn_agent = DdqnAgent(
        #     env.time_step_spec(),
        #     env.action_spec(),
        #     q_network = ddqn_network,
        #     optimizer = optimizer,
        #     td_errors_loss_fn = common.element_wise_squared_loss,
        #     train_step_counter = self.global_step)
        return env, agent
