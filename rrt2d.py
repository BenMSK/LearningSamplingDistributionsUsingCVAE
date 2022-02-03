#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# from random import randint
import math
import random

class RRT(object): # assume 2d holonomic robot
    def __init__(self, q_init, env, extend_len=1.0, goal_bias_ratio=0.05):
        self._root = q_init
        self._rrt = {q_init: q_init} # tree itself
        self._graph = {q_init: []}
        self._env = env
        self._extend_len = extend_len
        self._num_vertices = 1
        self._num_collision_checks = 0
        self.goal_bias_ratio = goal_bias_ratio

        # self.min_dist_to_goal = 999

    def search_nearest_vertex(self, p):
        rrt_vertices = self._rrt.keys()
        min_dist = (self._env.x_lims[1] - self._env.x_lims[0]) ** 2 + (self._env.y_lims[1] - self._env.y_lims[0]) ** 2
        min_q = tuple()
        for q in rrt_vertices:
            distance = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2
            if min_dist > distance:
                min_dist = distance
                min_q = q
        return min_q

    def is_contain(self, q):
        return q in self._rrt

    def add(self, q_new, q_near):
        self._rrt[q_new] = q_near # original
        self._graph[q_new] = [] # new for...
        self._graph[q_near].append(q_new)

    def remove(self, q):
        try:
            del self._rrt[q]
        except:
            print("wtf! Not in the tree")

    def get_rrt(self):
        return self._rrt

    def get_parent(self, q):
        return self._rrt[q]
    
    # def update_min(self, q, goal):
    #     if (self.min_dist_to_goal > self.distance(q, goal)):
    #         self.min_dist_to_goal = self.distance(q, goal)
    #         return True
    #     return False

    def extend(self, q_rand, add_node=True):
        # find nearest point in rrt
        q_near = self.search_nearest_vertex(q_rand)

        # calc new vertex
        q_new = self._calc_new_point(q_near, q_rand, delta_q=self._extend_len)
        # q_new = (int(q_new[0]), int(q_new[1])) # all descrete
        if self.is_collision(q_new) or self.is_contain(q_new) or (not self._env.collision_free_edge(q_near, q_new)):
            return None
        if add_node:
            self.add(q_new, q_near) # add(child, parent)
            self._num_collision_checks += 1
            self._num_vertices += 1
        return q_new

    def _calc_new_point(self, q_near, q_rand, delta_q=1.0):
        if self.distance(q_near, q_rand) < delta_q:
            return q_rand
        angle = math.atan2(q_rand[1] - q_near[1], q_rand[0] - q_near[0])
        q_new = (q_near[0] + delta_q * math.cos(angle), q_near[1] + delta_q * math.sin(angle))
        return q_new

    def is_collision(self, p):
        return not self._env.collision_free(p)

    def distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def is_goal_reached(self, q_new, end, goal_region_radius):
        if self.distance(q_new, end) <= goal_region_radius:
            return True
        else:
            return False

    def reconstruct_path(self, end):
        path = []
        q = end
        while not q == self._root:
            path.append([q, self.get_parent(q)])
            q = self.get_parent(q)
        # path.append(q)
        return path
    
    def get_solution_node(self, end):
        path = []
        q = end
        while not q == self._root:
            path.append(self.get_parent(q))
            q = self.get_parent(q)
        return path[:-1]

    def solve(self, goal, max_iter, goal_region_radius=3):
        iter = 0
        while iter <= max_iter:
            iter += 1
            if random.uniform(0, 1) > self.goal_bias_ratio:
                random_sample = (random.randint(self._env.x_lims[0], self._env.x_lims[1] - 1), random.randint(self._env.y_lims[0], self._env.y_lims[1] - 1))
                if self.is_collision(random_sample) or self.is_contain(random_sample):
                    continue
            else: # goal bias
                random_sample = goal
            q_new = self.extend(random_sample)
            if not q_new:
                continue

            if self.is_goal_reached(q_new, goal, goal_region_radius):
                # solution_path = self.reconstruct_path(q_new)
                return True
        return False