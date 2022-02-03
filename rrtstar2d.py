# -*- coding: utf-8 -*-
import math
from rrt2d import RRT as RRTBase

class RRTstar(RRTBase):
    def __init__(self, q_init, env, extend_len=1.0, dimension=2):
        super(RRTstar, self).__init__(q_init, env, extend_len=extend_len)
        self._dimension = dimension
        self._gamma = 2.0 * math.pow(1.0 + 1.0 / self._dimension, 1.0 / self._dimension) \
                         * math.pow(self._env.free_space_volume() / math.pi, 1.0 / self._dimension)
        # print("gamma: ", self._gamma)
        self._q_goal_set = []
        self._q_best = None
        self._best_cost = math.inf

    def cost(self, q):
        q_now = q
        c = 0
        while(q_now != self._root):
            c += self.distance(q_now, self.get_parent(q_now))
            q_now = self.get_parent(q_now)
        return c

    def rewire(self, q_new):
        r = min(self._extend_len, self._gamma * math.pow(math.log(self._num_vertices) / self._num_vertices, 1.0 / self._dimension))
        #print("r: ", r, self._gamma * math.pow(math.log(self._num_vertices) / self._num_vertices, 1.0 / self._dimension))
        q_near = [q for q in self._rrt.keys() if self.distance(q_new, q) <= r] #find near nodels
        for q in q_near: #ChooseParent
            if not self._env.collision_free_edge(q, q_new):
                continue
            if self.cost(q) + self.distance(q, q_new) < self.cost(q_new):
                self._rrt[q_new] = q
        
        for q in q_near: #Rewires a tree
            if q == self.get_parent(q_new): #already extended
                continue
            if not self._env.collision_free_edge(q, q_new):
                continue
            if self.cost(q_new) + self.distance(q_new, q) < self.cost(q):
                self._rrt[q] = q_new
    
    def update_best(self, goal):
        # print("q goal set: ", self._q_goal_set)
        for q in self._q_goal_set:
            new_cost = self.cost(q) + self.distance(q, goal)
            if new_cost < self._best_cost:
                self._q_best = q
                self._best_cost = new_cost


