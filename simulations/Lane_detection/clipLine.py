#! /usr/bin/env python3
# coding: UTF-8

import numpy as np

def clipLine(ln, p):
	corners = np.array([[p[0][0],p[1][0], p[2][0], p[3][0]],
 	                   [p[0][1],p[1][1], p[2][1], p[3][1]],
		               [1,1,1,1]])

	sides = np.zeros((3,4))
	sides[:, 0] = np.cross(corners[:, 0], corners[:, 1])
	sides[:, 1] = np.cross(corners[:, 1], corners[:, 2])
	sides[:, 2] = np.cross(corners[:, 2], corners[:, 3])
	sides[:, 3] = np.cross(corners[:, 3], corners[:, 0])
	
	intersections = np.zeros((3,4))
	intersections[:, 0] = np.cross(ln, sides[:, 0])  # Top
	intersections[:, 1] = np.cross(ln, sides[:, 1])  # Right
	intersections[:, 2] = np.cross(ln, sides[:, 2])  # Bottom
	intersections[:, 3] = np.cross(ln, sides[:, 3]) # Left
	for i in range(4):
		if intersections[2,i] != 0.:
			intersections[:,i] /= intersections[2,i]

	where = [False] * 4
	if intersections[0, 0] >= corners[0, 0] and intersections[0, 0] < corners[0, 1]:
		where[0] = True # TOP
	if intersections[1, 1] >= corners[1, 1] and intersections[1, 1] < corners[1, 2]:
		where[1] = True # RIGHT
	if intersections[0, 2] > corners[0, 3] and intersections[0, 2] <= corners[0, 2]:
		where[2] = True # BOTTOM
	if intersections[1, 3] > corners[1, 0] and intersections[1, 3] <= corners[1, 3]:
		where[3] = True # LEFT

	idx = [x for x in range(4) if where[x]]

	if len(idx) < 2:
		z = np.zeros(3)
		return z, z, False
	P1 = intersections[:, idx[0]].copy()
	P2 = intersections[:, idx[1]].copy()
	return P1, P2, True