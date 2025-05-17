#! /usr/bin/env python3
# coding: UTF-8

"""
clipLine.py
Author: Arturo Espinosa Romero
Date:  16/May/2025
"""

import numpy as np

class clipBox:	
	"""
		Defines a bounding box and provides a method to find the
  	     intersection points of an homogeneous line with it.

  	tl:	A two-element list, tuple or array that contain the coordinates
      	of the top-left corner of the bounding-box. The coordinate is
      	expected to be column-wise, i.e. [column, row].
  	wh: A two-element list, tuple or array that contain the width and height
      	of the bounding-box, e.g. [width, height].

	"""
	def __init__(self, tl, wh):
		"""
		Object constructor: Initializes the parameters of the bounding box.

		Parameters:
		:tl:	A two-element list, tuple or array that contain the coordinates
	    		of the top-left corner of the bounding-box. The coordinate is
	   			expected to be column-wise, i.e. [column, row].
	  :wh:	A two-element list, tuple or array that contain the width and height
	  			of the bounding-box, e.g. [width, height].
		"""
		self.corners = np.array([[tl[0], tl[0]+wh[0], tl[0]+wh[0],    tl[0]],
		                       [tl[1], tl[1],    tl[1]+wh[1], tl[1]+wh[1]],
		                       [    1,     1,              1,           1]])

		self.sides = np.zeros((3,4))
		self.sides[:, 0] = np.cross(self.corners[:, 0], self.corners[:, 1])
		self.sides[:, 1] = np.cross(self.corners[:, 1], self.corners[:, 2])
		self.sides[:, 2] = np.cross(self.corners[:, 2], self.corners[:, 3])
		self.sides[:, 3] = np.cross(self.corners[:, 3], self.corners[:, 0])

	def clipLine(self, ln):
		"""
		def clipLine(self, ln)

		Brief: computes the intersection points of an homogeneous line with
    	   the class bounding box.

		Parameters

	  :ln:	A three element numpy array that contains the coeficients that 
	   	  	define a line in the 2D-Plane, i.e. the vector [A,B,C], that
	     		corresponds to the line equation Ax+By+C=0.

		Returns
	  	P1, P2: Two numpy arrays that contain the homogeneous 2D coordinates of the
	    	      intersection of the line ln with the bounding-box defined by tl and wh.
	  	Status: A boolean which indicate if the intersections where found.
		"""
		intersections = np.zeros((3,4))
		intersections[:, 0] = np.cross(ln, self.sides[:, 0])  # Top
		intersections[:, 1] = np.cross(ln, self.sides[:, 1])  # Right
		intersections[:, 2] = np.cross(ln, self.sides[:, 2])  # Bottom
		intersections[:, 3] = np.cross(ln, self.sides[:, 3]) # Left
		for i in range(4):
			if intersections[2,i] != 0.:
				intersections[:,i] /= intersections[2,i]

		where = [False] * 4
		if intersections[0, 0] >= self.corners[0, 0] and intersections[0, 0] < self.corners[0, 1]:
			where[0] = True # TOP
		if intersections[1, 1] >= self.corners[1, 1] and intersections[1, 1] < self.corners[1, 2]:
			where[1] = True # RIGHT
		if intersections[0, 2] > self.corners[0, 3] and intersections[0, 2] <= self.corners[0, 2]:
			where[2] = True # BOTTOM
		if intersections[1, 3] > self.corners[1, 0] and intersections[1, 3] <= self.corners[1, 3]:
			where[3] = True # LEFT

		idx = [x for x in range(4) if where[x]]

		if len(idx) < 2:
			z = np.zeros(3)
			return z, z, False
		P1 = intersections[:, idx[0]].copy()
		P2 = intersections[:, idx[1]].copy()
		return P1, P2, True