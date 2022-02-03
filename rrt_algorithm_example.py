#!/usr/bin/env python
from random import random, randint
import random
import sys
sys.path.insert(0, "..")
import matplotlib.pyplot as plt
import time
import math
from env_2d import Env2D
import numpy as np
import os
from rrt2d import RRT
from rrtstar2d import RRTstar

planner = 'RRTstar'

#Step1: Set some problem parameters
x_lims = (0, 100) # low(inclusive), upper(exclusive) extents of world in x-axis
y_lims = (0, 100) # low(inclusive), upper(exclusive) extents of world in =y-axis
X_dimensions = np.array([x_lims, y_lims])
start = (2, 2)    #start state(world coordinates)
goal = (99, 99)  #goal state(world coordinates)
goal_bais_prob = 0.05
goal_region_radius = 3
tree_extend_length = 20

visualize = False

#Step 2: Load environment from file 
envfile = os.path.abspath("./motion_planning_datasets/100by100/shifting_gaps/train/1.png")
env_params = {'x_lims': x_lims, 'y_lims': y_lims}
planning_env = Env2D()
planning_env.initialize(envfile, env_params)

#Step 3: Initialize RRT
SMP = None
if planner == 'RRT':
    SMP = RRT(start, planning_env, extend_len=tree_extend_length)
    print("using RRT planner")
elif planner == 'RRTstar':
    SMP = RRTstar(start, planning_env, extend_len=tree_extend_length)
    print("using optimal-variant RRT (RRT*) planner")
else:
    print("Need a specific planner")
    quit()

solution_path = None
iter = 0
max_iter = 1500

#Step 4: Path planning algorithm starts!
while iter <= max_iter:
    iter += 1
    if random.random() > goal_bais_prob:
        random_sample = (random.uniform(x_lims[0], x_lims[1] - 1), random.uniform(y_lims[0], y_lims[1] - 1))
        if SMP.is_collision(random_sample) or SMP.is_contain(random_sample) or random_sample == start:
            continue
    else: # goal bias
        random_sample = goal
    q_new = SMP.extend(random_sample)
    if not q_new:
        continue

    if planner == 'RRT':
        if SMP.is_goal_reached(q_new, goal, goal_region_radius):
            solution_path = SMP.reconstruct_path(q_new)
            break

    elif planner == 'RRTstar':
        SMP.rewire(q_new)
        if SMP.is_goal_reached(q_new, goal, goal_region_radius):
            SMP._q_goal_set.append(q_new)
            print("goal!")

if SMP._q_goal_set != []:
    SMP.update_best(goal) # find best q
    solution_path = SMP.reconstruct_path(SMP._q_best)
    print("sol path: ", SMP.get_solution_node(SMP._q_best))
    print("total solution cost: ", SMP._best_cost)

if solution_path != None:
    planning_env.initialize_plot(start, goal, plot_grid=False)
    # planning_env.plot_tree(SMP.get_rrt(), 'dashed', 'blue', 0.5)
    planning_env.plot_path(solution_path, 'solid', 'red', 5)
    print("total vertices: ", len(SMP.get_rrt().keys()))
    plt.show()
else:
    print("planning fails...")
