#!/usr/bin/env python
""" @package environment_interface
Loads an environment file from a database and returns a 2D
occupancy grid.

Inputs : file_name, x y resolution (meters to pixel conversion)
Outputs:  - 2d occupancy grid of the environment
          - ability to check states in collision
"""
import random
import numpy as np
import math
import sys
from time import sleep
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
from scipy import ndimage
from utils import *

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def rgb2binary(rgb): # it converts binary image in rgb-form to binary image as itself.
    rgb = np.mean(rgb, axis=2)
    rgb[rgb!=0.0] = 1.0 # if it is obiously not an obstacle pixel(0), then it is regarded as free pixel(1)
    return rgb

class Env2D():
  def __init__(self):
    self.plot_initialized = False
    self.image = None

  def initialize(self, envfile, params):
    """Initialize environment from file with given params

      @param envfile - full path of the environment file
      @param params  - dict containing relevant parameters
                           {x_lims: [lb, ub] in x coordinate (meters),
                            y_lims: [lb, ub] in y coordinate (meters)}
      The world origin will always be assumed to be at (0,0) with z-axis pointing outwards
      towards right
    """
    try:
      self.image = plt.imread(envfile)
      self.image2 = plt.imread(envfile)# for visualizing alpha image
      self.image2[self.image2 != 0] = 1

      if len(self.image.shape) > 2:
        self.image = rgb2binary(self.image) # change the image to rgb
    except IOError:
      print("File doesn't exist. Please use correct naming convention for database eg. 0.png, 1.png .. and so on. You gave, %s"%(envfile))
    self.x_lims = params['x_lims']
    self.y_lims = params['y_lims']

    # resolutions
    self.x_res  = (self.x_lims[1] - self.x_lims[0])/((self.image.shape[1]-1)*1.)
    self.y_res  = (self.y_lims[1] - self.y_lims[0])/((self.image.shape[0]-1)*1.)

    orig_pix_x = math.floor(0 - self.x_lims[0]/self.x_res) #x coordinate of origin in pixel space
    orig_pix_y = math.floor(0 - self.y_lims[0]/self.y_res) #y coordinate of origin in pixel space
    self.orig_pix = (orig_pix_x, orig_pix_y)
  
  def free_space_volume(self):
    free_space_volume = 0
    for i in range(len(self.image)):
      for j in range(len(self.image[i])):
        free_space_volume += round(self.image[i][j])

    # print("free space volume: ", free_space_volume)
    return free_space_volume
    # round(self.image[pix_y][pix_x])

  def get_env_image(self):
    return self.image
  
  def set_env_image(self, image):
    self.image = image
    
  def set_params(self, params):
    self.x_lims = params['x_lims']
    self.y_lims = params['y_lims']

    # resolutions
    self.x_res  = (self.x_lims[1] - self.x_lims[0])/((self.image.shape[1]-1)*1.)
    self.y_res  = (self.y_lims[1] - self.y_lims[0])/((self.image.shape[0]-1)*1.)

    orig_pix_x = math.floor(0 - self.x_lims[0]/self.x_res) #x coordinate of origin in pixel space
    orig_pix_y = math.floor(0 - self.y_lims[0]/self.y_res) #y coordinate of origin in pixel space
    self.orig_pix = (orig_pix_x, orig_pix_y)

  def get_random_start_and_goal(self):
    random_start = tuple()
    random_goal = tuple()
    while True:
      random_start = (random.uniform(self.x_lims[0], self.x_lims[1] - 1), random.uniform(self.y_lims[0], self.y_lims[1] - 1))
      if self.collision_free(random_start):
        break
    while True:
      random_goal = (random.uniform(self.x_lims[0], self.x_lims[1] - 1), random.uniform(self.y_lims[0], self.y_lims[1] - 1))
      if self.collision_free(random_goal):
        break
    return (random_start, random_goal)

  def collision_free(self, state):
    """ Check if a state (continuous values) is in collision or not.

      @param state - tuple of (x,y) or (x,y,th) values in world frame
      @return 1 - free
              0 - collision
    """
    try:
      pix_x, pix_y = self.to_image_coordinates(state)
      return round(self.image[pix_y][pix_x])
    except IndexError:
      # print("Out of bounds, ", state, pix_x, pix_y)
      return 0

  def dist_between_points(self, a, b):
      """
      Return the Euclidean distance between two points
      :param a: first point
      :param b: second point
      :return: Euclidean distance between a and b
      """
      distance = np.linalg.norm(np.array(b) - np.array(a))
      return distance

  def es_points_along_line(self, start, end, r):
      """
      Equally-spaced points along a line defined by start, end, with resolution r
      :param start: starting point
      :param end: ending point
      :param r: maximum distance between points
      :return: yields points along line from start to end, separated by distance r
      """
      d = self.dist_between_points(start, end)
      n_points = int(np.ceil(d / r))
      if n_points > 1:
          step = d / (n_points - 1)
          for i in range(n_points):
              next_point = self.steer(start, end, i * step)
              yield next_point

  def steer(self, start, goal, d):
      """
      Return a point in the direction of the goal, that is distance away from start
      :param start: start location
      :param goal: goal location
      :param d: distance away from start
      :return: point in the direction of the goal, distance away from start
      """
      start, end = np.array(start), np.array(goal)
      v = end - start
      u = v / (np.sqrt(np.sum(v ** 2))) # (cos, sin)
      steered_point = start + u * d
      return tuple(steered_point)

  def collision_free_edge(self, state1, state2, r=1):
    """ Check if a state (continuous values) is in collision or not.

      @param state - tuple of (x,y) or (x,y,th) values in world frame
      @return 1 - free
              0 - collision
    """
    if (state1==state2):
      return True
    points = self.es_points_along_line(state1, state2, r)
    coll_free = all(map(self.collision_free, points))
    return coll_free

  def in_limits(self, state):
    """Filters a state to lie between the environment limits

    @param state - input state
    @return 1 - in limits
          0 - not in limits
    """
    pix_x, pix_y = self.to_image_coordinates(state)    
    if 0 <= pix_x < self.image.shape[1] and 0<= pix_y < self.image.shape[0] and self.x_lims[0] <= state[0] < self.x_lims[1] and self.y_lims[0] <= state[1] < self.y_lims[1]:
      return True
    return False

  def is_state_valid(self, state):
    """Checks if state is valid.

    For a state to be valid it must be within environment bounds and not in collision
    @param state - input state
    @return 1 - valid state
            0 - invalid state
    """
    if self.in_limits(state) and self.collision_free(state):
      return 1
    return 0
  
  def is_edge_valid(self, edge):
    """Takes as input an  edge(sequence of states) and checks if the entire edge is valid or not
    @param edge - list of states including start state and end state
    @return 1 - valid edge
            0 - invalid edge
            first_coll_state - None if edge valid, else first state on edge that is in collision
    """
    valid_edge = True
    first_coll_state = None
    for state in edge:
      print("state: ", state)
      if not self.in_limits(state):
        valid_edge = False
        break
      if not self.collision_free(state):
        valid_edge = False
        first_coll_state = state
        break
    return valid_edge, first_coll_state

  def to_image_coordinates(self, state):
    """Helper function that returns pixel coordinates for a state in
    continuous coordinates

    @param  - state in continuous world coordinates
    @return - state in pixel coordinates """
    pix_x = int(self.orig_pix[0] + math.floor(state[0]/self.x_res))
    pix_y = int(self.image.shape[1]-1 - (self.orig_pix[1] + math.floor(state[1]/self.y_res)))
    return (pix_x,pix_y)
  
  def to_world_coordinates(self, pix):
    """Helper function that returns world coordinates for a pixel

    @param  - state in continuous world coordinates
    @return - state in pixel coordinates """
    world_x = (pix[0] - self.orig_pix[0])*self.x_res 
    world_y = (pix[1] - self.orig_pix[0])*self.y_res   
    return (world_x, world_y)

  def get_env_lims(self):
    return self.x_lims, self.y_lims
  
  def get_obstacle_distance(self, state, norm=True):
    pix_x, pix_y = self.to_image_coordinates(state)
    if norm:
      d_obs = self.norm_edt[pix_y][pix_x] #distance to obstacle    
      obs_dx = self.dx_norm[pix_y][pix_x] #gradient in x direction 
      obs_dy = self.dy_norm[pix_y][pix_x] #gradient in y direction     
    else:
      d_obs = self.edt[pix_y][pix_x] #distance to obstacle    
      obs_dx = self.dx[pix_y][pix_x] #gradient in x direction 
      obs_dy = self.dy[pix_y][pix_x] #gradient in y direction 
    return d_obs, obs_dx, obs_dy

  def initialize_plot(self, start, goal, grid_res=None, plot_grid=False):
    # if not self.plot_initialized:
    self.figure, self.axes = plt.subplots()
    self.axes.set_xlim(self.x_lims)
    self.axes.set_ylim(self.y_lims)
    if plot_grid and grid_res:
      self.axes.set_xticks(np.arange(self.x_lims[0], self.x_lims[1], grid_res[0]))
      self.axes.set_yticks(np.arange(self.y_lims[0], self.y_lims[1], grid_res[1]))
      self.axes.grid(which='both')
    # self.figure.show() # if it is ON (un-commented), the plot figure is showed up and then disapp ear... 
    self.visualize_environment()
    self.line, = self.axes.plot([],[])
    self.background = self.figure.canvas.copy_from_bbox(self.axes.bbox) 
    self.plot_state(start, color='red', edge_color='white', msize=12)
    self.plot_state(goal, color=[0.06, 0.78, 0.78], edge_color='white', msize=12)
    self.figure.canvas.draw()
    self.background = self.figure.canvas.copy_from_bbox(self.axes.bbox) 
    self.plot_initialized = True

  def reset_plot(self, start, goal, grid_res=None):
    if self.plot_initialized:
      plt.close(self.figure) 
      self.initialize_plot(start, goal, grid_res)

  def visualize_environment(self):
    # if not self.plot_initialized:
        #// convert white pixels to transparent pixels
    alpha = ~np.all(self.image2 == 1.0, axis=2) * 255
    rgba = np.dstack((self.image2, alpha)).astype(np.uint8)
    self.axes.imshow(rgba, extent = (self.x_lims[0], self.x_lims[1], self.y_lims[0], self.x_lims[1]), cmap='gray', zorder=1)
    # self.axes.imshow(self.image, extent = (self.x_lims[0], self.x_lims[1], self.y_lims[0], self.x_lims[1]), cmap='gray')


  def plot_edge(self, edge, linestyle='solid', color='blue', linewidth=2):
    x_list = []
    y_list = []
    for s in edge:
      x_list.append(s[0])
      y_list.append(s[1])
    self.figure.canvas.restore_region(self.background)
    self.line.set_xdata(x_list)
    self.line.set_ydata(y_list)
    self.line.set_linestyle(linestyle)
    self.line.set_linewidth(linewidth)
    self.line.set_color(color)
    self.axes.draw_artist(self.line)
    self.figure.canvas.blit(self.axes.bbox)
    self.background = self.figure.canvas.copy_from_bbox(self.axes.bbox) 

  def plot_edges(self, edges,linestyle='solid', color='blue', linewidth=2):
    """Helper function that simply calls plot_edge for each edge"""
    for edge in edges:
      self.plot_edge(edge, linestyle, color, linewidth)

  def plot_state(self, state, color='red', edge_color='black', alpha=1.0, msize=9):
    """Plot a single state on the environment"""
    # self.figure.canvas.restore_region(self.background)
    self.axes.plot(state[0], state[1], marker='o',  markeredgecolor=edge_color, markersize=msize, color = color, alpha=alpha)
    self.figure.canvas.blit(self.axes.bbox)
    self.background = self.figure.canvas.copy_from_bbox(self.axes.bbox)
    
  def plot_path(self, path, linestyle='solid', color='blue', linewidth=2):
    flat_path = [item for sublist in path for item in sublist]
    self.plot_edge(flat_path, linestyle, color, linewidth)
  
  def plot_kde(self, tree, rx, ry, rz, random_sample):
    self.plot_tree(tree, 'dashed', 'blue', 1)
    # self.axes.scatter(rx, ry, color="pink", alpha=0.1)
    self.axes.scatter(rx, ry, c=rz, s=50, alpha=0.20, edgecolors='none')
    self.plot_state(random_sample, color='k', alpha=1.0, msize=7)
    self.figure.canvas.blit(self.axes.bbox)
    self.background = self.figure.canvas.copy_from_bbox(self.axes.bbox)

  def plot_pcolor(self, X, Y, Z, alpha=1.0, cmap='viridis'):
    self.axes.pcolor(X, Y, Z, alpha=alpha, cmap=cmap, shading='auto')

  def plot_title(self, name, fontsize=20):
    self.axes.set_title(name, fontdict = {'fontsize' : fontsize})

  def plot_current(self, tree, q_new, alpha=1.0):
    self.plot_tree(tree, 'dashed', 'blue', 1)
    self.plot_state(q_new, color='pink', alpha=alpha)

  def plot_tree(self, tree, linestyle='solid', color='blue', linewidth=1):
    for child, parent in tree.items(): # tree is a dictionary
      self.axes.plot((child[0], parent[0]), (child[1], parent[1]), marker='.', linestyle=linestyle, color=color, linewidth=linewidth)

  def plot_save(self, name):
    plt.savefig(name +'.png')
    plt.close(self.figure) 

  def close_plot(self):
    if self.plot_initialized:
      plt.close(self.figure)
      self.plot_initialized = False

  def clear(self):
    if self.plot_initialized:
      plt.close(self.figure)
      self.plot_initialized = False
    self.image = None
  
  def plot_states_torch(self, states, color='red', alpha=1.0, msize=9):
    for state in states:
      state = state_upscaling((state[0], state[1]), self.x_lims, self.y_lims)
      self.plot_state(state, color=color, alpha=alpha, msize=msize)    

  def plot_states(self, states, color='red', alpha=1.0, msize=9):
    for state in states:
      self.plot_state(state, color=color, alpha=alpha, msize=msize)    


  def plot_tree_torch(self, tree, linestyle='solid', color='blue', linewidth=1):
    tree = tree.cpu().numpy()
    for edge in tree:
      parent_x, parent_y, child_x, child_y = edge
      parent_x, parent_y = state_upscaling((parent_x, parent_y), self.x_lims, self.y_lims)
      child_x, child_y = state_upscaling((child_x, child_y), self.x_lims, self.y_lims)
      self.axes.plot((child_x, parent_x), (child_y, parent_y), marker='.', linestyle=linestyle, color=color, linewidth=linewidth)

  def plot_inference(self, q_start, q_goal, viz_tree, viz_label, viz_recon, plot_grid=False):
    self.initialize_plot(q_start, q_goal, plot_grid=plot_grid)
    self.plot_tree_torch(viz_tree)
    self.plot_states_torch(viz_recon, alpha=0.1)
    label = state_upscaling((viz_label[0], viz_label[1]), self.x_lims, self.y_lims)
    self.plot_state(label, color='lime', alpha=1.0)    
    
