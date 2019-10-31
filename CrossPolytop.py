# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:47:26 2019

@author: Григорий
"""
import numpy as np
from math import sqrt, factorial
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from time import time, ctime

#random.seed(43)
random.seed()

def saveExperiment(config_s, config_f, steps, name = None):

    """
    Saves result of one experiment to a new folder called Experiment_date.
    Namely, saves two sets of points -- config_s, config_f (objects of class Points), 
    which are starting and final configurations respectively and some additional parameters: 
    - Name = name;
    - Dimension;
    - Number of points;
    - Number of steps = steps;
    - Volume of final configuration.
    """

    if name is None:
        name = str(ctime(time()).replace(':', '_'))
    name_of_folder = 'Experiment_' + name
    os.makedirs(name_of_folder)
    F = open(name_of_folder + '/description.txt', 'w')
    F.write('Name = ' + name + '\n')
    F.write('Dimension = ' + str(config_s.dim) + '\n')
    F.write('Number of points = ' + str(config_s.num) + '\n') 
    F.write('Number of steps = ' + str(steps) + '\n')
    F.write('Volume = ' + str(config_f.getVolume()) + '\n')
    F.close()
    np.save(name_of_folder + '/coord_start.npy', config_s.coord)
    np.save(name_of_folder + '/coord_final.npy', config_f.coord)

def plot2D(config):

    """
    Plots a 2-dimensional set of points config (object of class Points). If dimension of 
    config is not 2, raises ValueError. For convenience also plots a unit circle.
    """

    if config.dim != 2:
        raise ValueError('Wrong dimension, must be 2!')
    plt.figure(figsize = (7, 7))
    for n, v in enumerate(config.coord):
        plt.scatter(v[0], v[1], label = str(n))
    arr1 = [(x-10000) / 10000 for x in range(20001)]
    arr2 = [sqrt(1 - ((x - 10000) / 10000) ** 2) for x in range(20001)]
    plt.plot(arr1, arr2, color = 'black', linewidth = 0.5)
    arr2 = [ - sqrt(1 - ((x - 10000) / 10000) ** 2) for x in range(20001)]
    plt.plot(arr1, arr2, color = 'black', linewidth = 0.5)
    plt.legend()
    plt.show()

def plot3D(config):

    """
    Plots a 3-dimensional set of points config (object of class Points). If dimension of 
    config is not 3, raises ValueError.
    """

    if config.dim != 3:
        raise ValueError('Wrong dimension, must be 3!')
    fig = plt.figure(figsize = (7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(0, 0, 0, color = 'black', label = 'center')
    ax.set_xlim3d(-1.1, 1.1)
    ax.set_ylim3d(-1.1, 1.1)
    ax.set_zlim3d(-1.1, 1.1)
    for n, v in enumerate(config.coord):
        ax.scatter(v[0], v[1], v[2], label = str(n))
    plt.legend()
    plt.show()

def genRandVector(dim = 3):

    """
    Generates a random vector uniformly on a unit sphere of dimension dim.
    """

    v = []
    for _ in range(dim):
        v.append(random.gauss(0, 1))
    v = np.array(v)
    v /= np.linalg.norm(v)
    return v

def getHyperplane(points):

    """
    Returns normal vector and distance to zero for a hyperplane, passing 
    through all points in Points, if number of points is appropriate and such 
    hyperplane is uniq. Normal vector is always oriented towards the halfspace, 
    which does not contain zero.
    """

    dim = len(points[0])
    M = np.array(points)
    one = np.array([1] * dim)
    try:
        normal = np.dot(one, np.linalg.inv(M.T))
    except np.linalg.LinAlgError:
        return None, None
    dist = 1 / np.linalg.norm(normal)
    normal *= dist
    if abs(np.dot(normal, points[0]) - dist) > 0.0000000001:
        normal *= -1
    return dist , normal


class Subsets(object):

    """
    The aim of this class is to generate all subsets of cardinality sub of a set of 
    cardinality num in some fixed order (alphabetical, but in fact it does not matter), 
    starting with the subset cur.
    """

    def __init__(self, num = 6, sub = 3, cur = None):

        """
        Initiates the first subset with cur, or with alphabetically first, 
        if cur is not determined.
        """

        if sub > num:
            raise ValueError("Subset can't be bigger than set!")
        self.num = num
        self.sub = sub
        if cur is None:
            cur = [n for n in range(sub)]
        self.cur = cur

    def next(self):

        """
        Updates cur to the next subset and returns it, if possible. 
        If cur is the last subset, returns None.
        """

        if self.cur == [n for n in range(self.num - self.sub, self.num)]:
            self.cur = None
            return None
        firstFree = self.sub - 1
        while self.cur[firstFree] == self.num + firstFree - self.sub:
            firstFree -= 1
        cur = self.cur[:firstFree] + [n for n in range(self.cur[firstFree] + 1,
                       self.cur[firstFree] + 1 + self.sub - firstFree)]
        self.cur = cur
        return self.cur

        
class Points(object):

    """
    Object of this class contains a set of num points on a unit sphere in Euclidian 
    space of dimension dim. It is desined to store the coordinates of points in a list 
    coord, calculate the set of the faces of its convex hull and store it into faces, 
    and also make some additional computations, such as calculating the volume of a convex
    hull or making a gradient discent with respect to this volume.
    """

    def __init__(self, dim = 3, num = 6, coord = None, faces = None):

        """
        Initiates the set of num points on a unit sphere in Euclidian 
        space of dimension dim with the set coord, or with a random set 
        of independent uniformly distributed points on a unit sphere. 
        Then calculates the set of faces of its convex hull, if it is not 
        given as an argument, and stores it into self.faces.
        """

        self.dim = dim
        self.num = num
        if self.num < self.dim:
            raise ValueError("Not enough points!")
        if coord is None:
            coord = []
            for i in range(num):
                coord.append(genRandVector(dim))
        self.coord = coord
        if faces is None:
            faces = self.getFaces()
        self.faces = faces

    def isFace(self, pointsNum):

        """
        Checks if a subset pointsNum of a set of all points in self is a face 
        of a convex hull.
        """

        dist, normal = getHyperplane([self.coord[index]\
                                      for index in pointsNum])
        if dist is None:
            return False
        sign = 0
        for num, point in enumerate(self.coord):
            if num not in pointsNum:
                if sign * (np.dot(point, normal) - dist) >= 0:
                    sign = (np.dot(point, normal) - dist)
                else:
                    return False
        return True

    def split(self, pointsNum):

        """
        Splits the set of all points in self into two, which are subsets of 
        the halfspaces, bounded by a hyperplane, containing points from pointsNum.
        """

        dist, normal = getHyperplane([self.coord[index]\
                                      for index in pointsNum])
        set0 = pointsNum.copy()
        set1 = pointsNum.copy()
        for num, point in enumerate(self.coord):
            if (np.dot(point, normal) - dist) >= 0 and num not in set1:
                set1.append(num)
            elif num not in set0:
                set0.append(num)
        set0.sort()
        set1.sort()
        return set0, set1
    
    def getFaces(self, pointsNum = None):

        """
        Calculates the set of faces of a convex hull of points in pointsNum, 
        or of all points in self, if pointsNum is None.
        """

        if pointsNum is None:
            pointsNum = [n for n in range(self.num)]
        if len(pointsNum) == self.dim and self.isFace(pointsNum):
            return [pointsNum]
        subsets = Subsets(len(pointsNum), self.dim)
        faces = []
        while subsets.cur is not None:
            s = [pointsNum[n] for n in subsets.cur]
            if self.isFace(s):
                faces.append(s)
            subsets.next()
        return faces
    
    def getVolume(self):

        """
        Calculates the volume of the convex hull of all points in self.
        """

        v = 0
        for face in self.faces:
            cur_vol = abs(np.linalg.det(np.array([self.coord[index]\
                                                  for index in face])))
            n = 0
            while n in face:
                n += 1
            dist, normal = getHyperplane([self.coord[index] for index in face])
            if np.dot(self.coord[n], normal) - dist > 0:
                v -= cur_vol
            else:
                v += cur_vol
        return v / factorial(self.dim)
    
    def calculateGrad(self, index, pres = 0.0001):

        """
        Calculates the direction of the gradient of volume of convex hull 
        of all points in self (returns a unit vector) with respect to the 
        position of point with number index. Parameter pres corresponds to 
        the precision of this calculation.
        """

        grad = []
        for pos in range(self.dim):
            point_apd = self.coord[index] + np.array([0] * pos + [pres] +
                                              [0] * (self.dim - pos - 1))
            v = - self.getVolume()
            config = Points(self.dim, self.num,
                    self.coord[:index] + [point_apd] + self.coord[index + 1:])
            v += config.getVolume()
            grad.append(v)
        grad = np.array(grad)
        return grad / np.linalg.norm(grad)
    
    def updatePositions(self, step = 0.1, pres = 0.0001):

        """
        Updates positions of points in self, moving them in the direction of 
        an appropriate gradient of volume by length step. Parameter pres 
        corresponds to the precision of the calculation of gradient.
        """

        new_coord = []
        for index in range(self.num):
            point_apd = self.calculateGrad(index, pres) * step +\
                        self.coord[index]
            point_apd /= np.linalg.norm(point_apd)
            new_coord.append(point_apd)
        config = Points(self.dim, self.num, new_coord)
        return config

    def distToCrosspolytop(self):

        """
        Estimates pointwise distance to the closest crosspolytop.
        """

        if len(self.coord) != 2 * self.dim:
            return None
        crosspolytop = []
        for num in range(self.dim):
            u = self.coord[self.faces[0][num]]
            for v in crosspolytop:
                u -= np.dot(v, u) * v
            u /= np.linalg.norm(u)
            crosspolytop.append(u)
        dist = 0
        for v in self.coord:
            m = 10000
            for u in crosspolytop:
                d = min(np.linalg.norm(u - v), np.linalg.norm(u + v))
                if d < m:
                    m = d
            dist += m ** 2
        return dist
