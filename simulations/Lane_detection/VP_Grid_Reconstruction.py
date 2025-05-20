#! /usr/bin/env python3
# coding: UTF-8

import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from clipBox import *
import datetime

def null_space(A, rcond = None):
      """
      Computes the null space of a matrix.
      """
      u, s, vh = np.linalg.svd(A, full_matrices=True)
      M, N = u.shape[0], vh.shape[1]
      if rcond is None:
         rcond = np.finfo(s.dtype).eps * max(M, N)
      tol = np.amax(s) * rcond
      num = np.sum(s > tol, dtype=int)
      Q = vh[num:, :].T.conj()
      return Q
   
def norm_points(p):
   r,c = p.shape
   for i in range(c):
      if p[-1,i] != 0:
         p[:,i] /= p[-1,i]
   return p

def time_difference(start_time, end_time):
   """
   Calculates the time difference between two datetime objects in microseconds.
   """
   delta = end_time - start_time
   if delta.seconds == 0:
      return delta.microseconds
   else:
      return delta.seconds*1e6+microseconds

def sortPts(P):
   theta = np.zeros(4)
   mX = np.mean(P[0,: ])
   mY = np.mean(P[1, :])
   for k in range(4):
      Dx = P[0, k] - mX
      Dy = P[1, k] - mY
      theta[k] = np.arctan2(Dy, Dx)
   
   indices = sorted(range(len(theta)), key = lambda index: theta[index])
   sP = P[:, indices]
   return sP, indices

def sortPtsIdx(p,i,j):
   P = np.array([[p[i, 0, 0],p[i, 0, 2],p[j, 0, 0],p[j, 0, 2]],
              [p[i, 0, 1],p[i, 0, 3],p[j, 0, 1],p[j, 0, 3]],
              [  1,            1,       1,      1]]).astype('float64')
   return sortPts(P)

def line_similarity(line_a, line_b, threshold=1, normType = 0, clpB=None):
   distance = np.inf
   pts = None
   tstamps = []
   tstamps.append([0,datetime.datetime.now()])
   if normType == 1:
     # Normalize the lines as homogeneous variable
     linea_n = line_a / line_a[2]
     lineb_n = line_b / line_b[2]
     tmp = linea_n[:2] - lineb_n[:2]
     distance = np.dot(tmp, tmp)
     tstamps.append([1, datetime.datetime.now()])
       
   elif normType == 2:
     # Normalize the lines according to their size
     lineb_n = line_b / np.linalg.norm(line_b)
     linea_n = line_a / np.linalg.norm(line_a)
     tmp = lineb_n - linea_n
     distance = np.dot(tmp, tmp)
     tstamps.append([2, datetime.datetime.now()])
   elif normType == 3:
     if clpB == None:
       clpB = clipBox((0,540),(1920,540))
     pl1, pl2, success = clpB.clipLine(line_a)
     if success == True:
       tstamps.append([2, datetime.datetime.now()])
       pl3, pl4, success = clpB.clipLine(line_b)
       if success == True:
         mid1 = np.array([(pl1[0] + pl2[0]) / 2, (pl1[1] + pl2[1]) / 2])
         mid2 = np.array([(pl3[0] + pl4[0]) / 2, (pl3[1] + pl4[1]) / 2])
         d = np.linalg.norm(mid1 - mid2) ** 2
         return d < threshold * threshold, d
     else:
       if clpB == None:
         clpB = clipBox((0,540),(1920,540))
       pl1, pl2, success = clpB.clipLine(line_a)
       if success == True:
         tstamps.append([2, datetime.datetime.now()])
         pl3, pl4, success = clpB.clipLine(line_b)
         if success == True:
            tstamps.append([3, datetime.datetime.now()])
            P = np.hstack([pl1,pl2,pl3,pl4]).reshape(4,3).transpose()
            tstamps.append([4, datetime.datetime.now()])
            sP, idx = sortPts(P)
            tstamps.append([5, datetime.datetime.now()])
            d=[]
            tmp = sP[:2,0]-sP[:2,1]
            d.append(np.dot(tmp, tmp)) #Squared Distance between P[0,:] and P[1,:]
            tmp = sP[:2,2]-sP[:2,3]
            d.append(np.dot(tmp, tmp)) #Squared Distance between P[0,:] and P[1,:]
            pts = P.copy()
            tstamps.append([6, datetime.datetime.now()])
            if d[0] > d[1]:
              distance = d[0]
              pts = np.hstack([pts, np.array(sP[:,0],ndmin=2).transpose()])
              pts = np.hstack([pts, np.array(sP[:,1],ndmin=2).transpose()])
            else:
              distance = d[1]
              pts = np.hstack([pts, np.array(sP[:,2],ndmin=2).transpose()])
              pts = np.hstack([pts, np.array(sP[:,3],ndmin=2).transpose()])
            tstamps.append([7, datetime.datetime.now()])

   # Compute similarity
   tstamps.append([8, datetime.datetime.now()])
   
   for i in range(1,len(tstamps)):
     tstamps[i][1] = time_difference(tstamps[0][1],tstamps[i][1])
   return distance <= (threshold * threshold), distance, pts, tstamps

def distance_between_lines(line1, line2, img_width, img_height):
   """
   Calculate distance between two lines by extending them to image boundaries
   and computing the distance between their midpoints.

   Args:
      line1: First line (x1, y1, x2, y2)
      line2: Second line (x1, y1, x2, y2)
      img_width: Width of the image
      img_height: Height of the image

   Returns:
      Euclidean distance between the midpoints of the extended lines
   """
   # Get extended endpoints for both lines
   (x1_1, y1_1), (x2_1, y2_1) = get_line_endpoints(line1[0], img_width, img_height)
   (x1_2, y1_2), (x2_2, y2_2) = get_line_endpoints(line2[0], img_width, img_height)

   # Calculate midpoints of extended lines
   mid1 = np.array([(x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2])
   mid2 = np.array([(x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2])

   # Return Euclidean distance between midpoints
   return np.linalg.norm(mid1 - mid2)

def select_lines_with_distance_orig(lines_vp1, img_width, img_height, threshold_min, threshold_max, max_attempts=1000):
   """
   Select two random lines with distance between thresholds.

   Args:
      lines_vp1: List of lines, each as (x1, y1, x2, y2)
      img_width: Image width for line extension
      img_height: Image height for line extension
      threshold_min: Minimum allowed distance between lines
      threshold_max: Maximum allowed distance between lines
      max_attempts: Maximum attempts before giving up

   Returns:
      Tuple of (idx1, idx2, distance) for the selected lines

   Raises:
      ValueError if no suitable pair is found
   """
   n = len(lines_vp1)
   if n < 2:
      raise ValueError("Need at least 2 lines to select a pair")

   for _ in range(max_attempts):
      # Select two distinct random indices
      idx1, idx2 = random.sample(range(n), 2)
      line1 = lines_vp1[idx1]
      line2 = lines_vp1[idx2]

      # Calculate distance between extended lines
      distance = distance_between_lines(line1, line2, img_width, img_height)

      # Check if distance is within desired range
      if threshold_min <= distance <= threshold_max:
         return idx1, idx2, distance

   raise ValueError(f"No valid line pair found after {max_attempts} attempts")

def select_lines_with_distance(lines_vp1, clpBox, threshold_min, threshold_max, max_attempts=1000):
   """
   Select two random lines with distance between thresholds.

   Args:
      lines_vp1: List of lines, each as (x1, y1, x2, y2)
      img_width: Image width for line extension
      img_height: Image height for line extension
      threshold_min: Minimum allowed distance between lines
      threshold_max: Maximum allowed distance between lines
      max_attempts: Maximum attempts before giving up

   Returns:
      Tuple of (idx1, idx2, distance) for the selected lines

   Raises:
      ValueError if no suitable pair is found
   """
   n = len(lines_vp1)
   if n < 2:
      raise ValueError("Need at least 2 lines to select a pair")

   threshold_min = float(threshold_min) ** 2
   threshold_max = float(threshold_max) ** 2
   print ("Select_Lines_with_distance", flush=True)
   for _ in range(max_attempts):
      # Select two distinct random indices
      idx1, idx2 = random.sample(range(n), 2)
      
      line1 = lines_vp1[idx1][0]
      M = np.array([[line1[0], line1[1], 1], [line1[2], line1[3], 1]])
      l1 = null_space(M)[:, 0]
      l1 /= l1[2]
      
      line2 = lines_vp1[idx2][0]
      M = np.array([[line2[0], line2[1], 1], [line2[2], line2[3], 1]])
      l1 = null_space(M)[:, 0]
      l1 /= l1[2]
      
      # Calculate distance between extended lines
      #_, distance, _, tstamps = line_similarity(l1, l2, threshold=1, normType = 0, clpB=clpBox)
      _, distance =  line_similarity(l1, l2, threshold=1, normType = 3, clpB=clpBox)
      # Check if distance is within desired range   
      if threshold_min <= distance <= threshold_max:
         print ("Se encontró que las lineas %d y %d son similares (%f)"% (idx1, idx2, distance),flush=True) 
         return idx1, idx2, distance

   raise ValueError(f"No valid line pair found after {max_attempts} attempts")

def line_passes_near_vp(line, vp, threshold=10):
   x1, y1, x2, y2 = line
   line_params = np.polyfit([x1, x2], [y1, y2], 1)
   slope, intercept = line_params
   vp_x, vp_y = vp[0], vp[1]
   return abs(vp_y - (slope * vp_x + intercept)) < threshold

def def_grid_lines(r0,r1,c0,c1,w,h):
   R=np.linspace(r0,r1,r1-r0+1)
   C=np.linspace(c0,c1,c1-c0+1)
   n = len(R)+len(C)
   l=np.zeros((3, n))
   
   #Definimos primero lineas horizontales
   idx=0
   for i in R:
      l[:,idx]=[0, 1, h*i]
      idx +=1
      
   #Definimos primero lineas verticales
   for i in C:
      l[:,idx]=[1, 0, h*i]
      idx=idx+1
   return l

# Definir puntos de fuga y puntos
vp1 = np.array([-3, 540, 1])
vp2 = np.array([1916, 540, 1])

# Load data
lines_near_vps = np.load('merged_lines_near_vps.npy')

clpBox = clipBox((0,0),(1920,1080))

lines_vp1 = []
lines_vp2 = []

img = cv2.imread('vp.jpg')
cv2.circle(img, (vp1[0], vp1[1]), 5, (0, 255, 0), -1)
cv2.circle(img, (vp2[0], vp2[1]), 5, (0, 255, 0), -1)

for line in lines_near_vps:
   if line_passes_near_vp(line[0], vp1):
      lines_vp1.append(line)
   elif line_passes_near_vp(line[0], vp2):
      lines_vp2.append(line)

# Draw lines passing through vp1
for line in lines_vp1:
   x1, y1, x2, y2 = line[0]
   M = np.array([[x1, y1, 1], [x2, y2, 1]])
   l1 = null_space(M)[:, 0]
   
   pt1, pt2, success = clpBox.clipLine(l1)
   if success:
      pt1 /= pt1[2]
      pt2 /= pt2[2]
      
      pt1=tuple(np.round(pt1[:2]).astype('int64'))
      pt2=tuple(np.round(pt2[:2]).astype('int64'))
   
      cv2.line(img, pt1, pt2, (255, 0, 0), 1)  # Blue color for lines passing through vp1

# Draw lines passing through vp2
for line in lines_vp2:
   x1, y1, x2, y2 = line[0]
   M = np.array([[x1, y1, 1], [x2, y2, 1]])
   l2 = np.array(null_space(M)[:, 0])
   pt1, pt2, success = clpBox.clipLine(l2)
   if success:
      pt1 /= pt1[2]
      pt2 /= pt2[2]
   
      pt1=tuple(np.round(pt1[:2]).astype('int64'))
      pt2=tuple(np.round(pt2[:2]).astype('int64'))
   
      cv2.line(img, pt1, pt2[:2], (0, 0, 255), 1)  # Red color for lines passing through vp2
   
# Display the image with the drawn lines
cv2.namedWindow("Image with Lines", cv2.WINDOW_NORMAL)
cv2.imshow('Image with Lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

def points_to_homogeneous_line(x1, y1, x2, y2):
   return np.cross([x1, y1, 1], [x2, y2, 1])

def order_lines_by_intersection(lines, horizon_y=540, offset=10):
   H10 = np.array([0, 1, -(horizon_y + offset)])  # Línea horizontal

   intersections = []
   for line in lines:
      intersection = np.cross(line, H10)
      intersection = intersection / intersection[-1]  # Normalizar
      intersections.append(intersection)

   # Ordenar las líneas según la coordenada x de las intersecciones
   sorted_indices = np.argsort([pt[0] for pt in intersections])  # Ordenar por x
   ordered_lines = [lines[i] for i in sorted_indices]
   return ordered_lines

def main_process(lines_near_vps, vp1, vp2, thr=1, iterations=100, grid_size=5, region=[1920, 1080]):
   results = []
   
   lines_vp1 = []
   lines_vp2 = []


   for line in lines_near_vps:
    
      if line_passes_near_vp(line[0], vp1):
         lines_vp1.append(line)
      elif line_passes_near_vp(line[0], vp2):
         lines_vp2.append(line)
   

   #Define Grid Lines
   grid_lines = def_grid_lines(-grid_size, grid_size, -grid_size, grid_size, 1, 1)
   clpBox = clipBox((0,0), (region[0], region[1]))
   
   # Define the square of 1x1
   pr = np.ones((3,4))
   pr[:2,0] = [0., 1]
   pr[:2,1] = [0., 0.]
   pr[:2,2] = [1, 0.]
   pr[:2,3] = [1, 1.]
   pr = norm_points(pr)
   src_pts = np.float32([pr[:, 0], pr[:, 1], pr[:, 2], pr[:, 3]])
   
   for it in range(iterations):
      # Step 1: Select random lines
      
      # Select two random lines from lines_vp1
      idx1, idx2, distance = select_lines_with_distance(lines_vp1, clpBox, 50, 100)
      random_lines_vp1 = [lines_vp1[idx1], lines_vp1[idx2]]
      
      # Select two random lines from lines_vp2
      idx1, idx2, distance = select_lines_with_distance(lines_vp2, clpBox, 50, 100)
      random_lines_vp2 = [lines_vp2[idx1], lines_vp2[idx2]]
      homogeneous_lines = [points_to_homogeneous_line(*line[0]) for line in random_lines_vp1 + random_lines_vp2]
      # print(homogeneous_lines)
      
      # Step 2: Find intersections
      ordered_lines = order_lines_by_intersection(homogeneous_lines)

      l1 = ordered_lines[1]
      l2 = ordered_lines[0]
      l3 = ordered_lines[2]
      l4 = ordered_lines[3]

      p1 = np.cross(l2, l3)
      p2 = np.cross(l1, l3)
      p3 = np.cross(l1, l4)
      p4 = np.cross(l2, l4)

      p = np.zeros((3, 4))
      p[:, 0] = p1
      p[:, 1] = p2
      p[:, 2] = p3
      p[:, 3] = p4
      p = norm_points(p)

      #Step 3 Compute homography M
      
      dst_pts = np.float32([p[:, 0], p[:, 1], p[:, 2], p[:, 3]])
      M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)  

      try:
         Hl = np.linalg.inv(M).T
      except np.linalg.LinAlgError:
         continue

      projected_lines = np.dot(Hl, grid_lines)
      projected_lines = projected_lines / np.linalg.norm(projected_lines, axis=0)

      # Step 4: Compare lines
      similarities = 0
      Dist = 0
      clp = clipBox((0,region[1]//2),(region[0], region[1]//2))
      cont = 0
      for line in projected_lines.T:
         for original_line in lines_near_vps:
            cont+=1
            original_line =original_line[0]
            homogeneous_line = points_to_homogeneous_line(original_line[0], original_line[1], original_line[2], original_line[3])

            isSimil,d,_,_ = line_similarity(line, homogeneous_line, threshold = thr, normType = 0, clpB = clp)
            #isSimil,d,_ = line_similarity_old(line, homogeneous_line, 0.2)
            if isSimil:
               similarities += 1
               Dist += d
               break

      # Step 5: Save results
      if similarities > 0:
         results.append((similarities, Dist/similarities, M, it))

   # Step 6: Rank results
   results.sort(key=lambda x: x[0], reverse=True)
   return results

# Run the process
best_results = main_process(lines_near_vps, vp1, vp2, thr=5, iterations=100, grid_size=5, region=(1920,1080))
print ("best results:", best_results, flush=True)

# Print the top 5 results
for i, (similarities, d, M, it) in enumerate(best_results[:5]):
   print(f"Rank {i+1}: Similarities = {similarities}")
   print("Dist promedio: ", d)
   print("Iteracion: ", it)
   print("Matrix M:")
   print(M)

for i, (similarities, d, M, it) in enumerate(best_results[:5]):
   img = cv2.imread('vp.jpg')
   lines = def_grid_lines(-5,5,-5,5,1,1)
   Hl = np.linalg.inv(M).T
   projected_lines = np.dot(Hl, lines)
   projected_lines = projected_lines / np.linalg.norm(projected_lines, axis=0)
   # Dibujar líneas proyectadas
   for i in range(projected_lines.shape[1]):
      a, b, c = projected_lines[:, i]
      # Encontrar dos puntos en la línea (para dibujar)
      x0, y0 = 0, int(-c/b) if b != 0 else 0
      x1, y1 = img.shape[1], int((-c - a*img.shape[1])/b) if b != 0 else 0
      cv2.line(img, (x0, y0), (x1, y1), (255, 0, 0), 2)

   cv2.namedWindow("Projected Grid", cv2.WINDOW_NORMAL)
   cv2.imshow("Projected Grid", img)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
