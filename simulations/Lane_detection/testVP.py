#! /usr/bin/env python3
# encoding: utf-8

import os
import cv2
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from sklearn.cluster import AgglomerativeClustering

class ImageInfo:
   def __init__(self, image_path):
      self.image_path = image_path
      self.distances, self.angles, self.time = self.get_tags(image_path)

   def get_tags(self, image_path):
      distances = []
      angles = []
      time = 0
      filename = os.path.basename(image_path)
      filename = filename.split('_')
      for word in filename:
         if word.startswith('d'):
            distances.append(float(word[3:]))
         if word.startswith('a'):
            angles.append(float(word[3:]))
         if word.startswith('t'):
            time = float(word[2:])

      return distances, angles, time

class linesIntersections:
   def __init__(self, lInf, inter, interIdx, interInf, interInfIdx, img):
      self.linesInfo = lInf
      self.intersections = inter
      self.intLinesIdx = interIdx
      self.intersectionsInf = interInf
      self.intLineInfIdx = interInfIdx,
      self.image = img
      self.nLines, _ = self.linesInfo.shape
      self.nIntersections, _ = self.intersections.shape
      self.nIntersectionsInf, _ = self.intersectionsInf.shape

def loadTaggedImages(directory):
   images = []
   for filename in os.listdir(directory):
      if filename.endswith(".jpg") or filename.endswith(".png"):
         route = os.path.join(directory, filename)
         images.append(ImageInfo(route))
   return images

@staticmethod
def null_space(A, rcond=None):
   u, s, vh = la.svd(A, full_matrices=True)
   M, N = u.shape[0], vh.shape[1]
   if rcond is None:
      rcond = np.finfo(s.dtype).eps * max(M, N)
   tol = np.amax(s) * rcond
   num = np.sum(s > tol, dtype=int)
   Q = vh[num:, :].T.conj()
   return Q

@staticmethod
def areEqual(a, b, ord=7):
   if a == b:
      return True
   if a != 0. and b != 0.:
      val = np.abs((a - b) / max(np.abs([a, b])))
   else:
      val = max(np.abs([a, b]))
   return -np.log10(val) > ord

def cut_below_horizon(image):
   height, width = image.shape
   half_width = height // 2
   bottom_half = image.copy()
   bottom_half[0:half_width, :] = 0
   return bottom_half

def getLinesInfo(lines):
   linesInfo = np.zeros((len(lines), 8))
   if lines is not None:
      for i, line in enumerate(lines):
         x1, y1, x2, y2 = line[0]
         M = np.array([[x1, y1, 1], [x2, y2, 1]])
         linesInfo[i, :3] = null_space(M)[:, 0]
         linesInfo[i, 3] = np.sqrt((x1-x2)**2+(y1-y2)**2)
         linesInfo[i, 4:] =x1, y1, x2, y2
   return linesInfo

def computeIntersections(linesInfo):
   nLines = len(linesInfo)
   nComb = ((nLines * (nLines - 1)) // 2)
   pn =  np.zeros((nComb, 3)) #Aqui almacenamos las lineas que no estan en el infinito
   pInf = np.zeros((nComb, 3)) #Aqui almacenamos las lineas que si estan en el infinito
   intLinesIdx = np.zeros((nComb, 2), dtype='int32')
   intLinesInfIdx = np.zeros((nComb, 2), dtype='int32')
   idx  = 0
   idxInf = 0
   for i in range(nLines - 1):
      for j in range(i + 1, nLines):
         homoP = np.cross(linesInfo[i, :3], linesInfo[j, :3])
         if not areEqual(homoP[2], 0., 8):
            homoP /= homoP[2]
            pn[idx, :] = homoP
            intLinesIdx[idx, :] = [i, j]
            idx += 1
         else:
            pInf[idxInf, :] = homoP
            intLinesInfIdx[idxInf, :] = [i, j]
            idxInf += 1
   return pn[:idx, :], intLinesIdx[:idx, :], pInf[:idxInf, :], intLinesInfIdx[:idxInf, :]

def getIntersections(image, threshold=210):
   imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   bottom_half = cut_below_horizon(imageGray)
   _, binary_gray_image = cv2.threshold(bottom_half, threshold, 255, cv2.THRESH_BINARY)
   edges = cv2.Canny(binary_gray_image, 100, 200)
   lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=50, maxLineGap=40)
   linesInfo = getLinesInfo(lines)
   intersections, intLinesIdx, IntersectionsInf, intLineInfIdx = computeIntersections(linesInfo)
   L = linesIntersections(linesInfo, intersections, intLinesIdx, IntersectionsInf, intLineInfIdx, image)
   return L

def drawLine(img, L, color=(0,0,0), width=1):
    x1, y1, x2, y2 = int(L[4]), int(L[5]), int(L[6]), int(L[7])
    cv2.line(img, (x1, y1), (x2, y2), color, width)

def drawWholeLine(img, L, color, widht):
   r, c, _ = img.shape

   linEq = L[:3].copy()
   linLeft   = np.array([1, 0,   0])
   linRight  = np.array([1, 0, 1-c])
   linTop    = np.array([0, 1, 0])
   linBottom = np.array([0, 1, 1-r])
   intL = np.cross(linEq, linLeft)
   intL = intL / intL[3]
   intL = np.cross(linEq, linRight)
   intR = intR / intR[3]
   intT = np.cross(linEq, linTop)
   intT = intT / intT[3]
   intB = np.cross(linEq, linBottom)
   intB = intB / intB[3]

def exploreIntersections(image, winName, intInfo):
   flag = True
   idx = 0
   while flag == True:
      img = image.copy()

      for L in intInfo.linesInfo[:]:
         drawLine(img, L, (0,255,255), 1)
      for I in intInfo.intersections[:]:
         x, y = int(I[0]), int(I[1])
         cv2.circle(img, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
      i, j = intInfo.intLinesIdx[idx]
      drawLine(img, intInfo.linesInfo[i], (255,255, 0), 2)
      drawLine(img, intInfo.linesInfo[j], (255,255, 0), 2)
      print(intInfo.linesInfo[i,:])
      
      l1 =  intInfo.linesInfo[i,:3].copy()
      l2 =  intInfo.linesInfo[j,:3].copy()
      l1 *= 100
      l2 *= 100
      print("L1 = %6.3fx+%6.3fy+%6.3fx=0" % (l1[0], l1[1], l1[2]))
      print("L2 = %6.3fx+%6.3fy+%6.3fx=0\n" % (l2[0], l2[1], l2[2]))
            
      x, y = intInfo.intersections[idx,:2]
      x, y = int(x), int(y)
      cv2.circle(img, (x, y), radius=10, color=(255, 255, 255), thickness=-1)
      
      cv2.imshow(winName, img)
      val = cv2.waitKey(0)
      if val == 27:
         flag = False
      else:
         if chr(val)=='a' and idx > 0:
            idx -= 1
         elif  chr(val)=='d' and idx + 1 < intInfo.nIntersections:
            idx += 1

def compute_VP(inter):
   data=inter.intersections[:,:2]
   AG = AgglomerativeClustering(distance_threshold=0.5, n_clusters=None, compute_full_tree=True).fit(data)
   nLabels = max(AG.labels_)
   print ("leaves       = ", AG.n_leaves_)
   print ("num Features = ", AG.n_features_in_) 
   print("labels        = ", nLabels)
  
   clusters=[]
   for i in range(nLabels):
      leaves = [j for j in range(nLabels) if AG.labels_[j]==i]
      clusters.append(leaves)
   clustersSize = [len(j) for j in clusters]
   print("clusters:", clusters)
   print("clustersSize:", clustersSize)
   C=clustersSize.copy()
   C.sort()
   print("clustersSize Ordenado:",C)
   print(max(clustersSize))
   print([k for k in range(len(clustersSize)) if clustersSize[k]==3])
#--------------------------------------------------------------------

showVideo = False

if __name__ == "__main__":
   images_info = loadTaggedImages('../manual_sequence/sec4/')
    
   images_info.sort(key=lambda x: x.time, reverse=False)

   if showVideo == True:
      cv2.namedWindow("Imagenes", cv2.WINDOW_NORMAL)

      for idx in range(len(images_info)):
         image = cv2.imread(images_info[idx].image_path, cv2.IMREAD_COLOR)
         cv2.imshow("Imagenes", image)
         val = cv2.waitKey(30)
         if val == 27:
            break

      cv2.destroyWindow("Imagenes")

#---------

   cv2.namedWindow("Lineas", cv2.WINDOW_NORMAL)
   
   for idx in range(len(images_info)):
      image = cv2.imread(images_info[idx].image_path, cv2.IMREAD_COLOR)

      intersectionsInfo = getIntersections(image)
      print ("lines found: ", intersectionsInfo.nLines)
      print ("intersections found: ", intersectionsInfo.nIntersections)
      print ("intersections Infinity found: ", intersectionsInfo.nIntersectionsInf)
      print ("-"*80, "\n")

      exploreIntersections(image, "Lineas", intersectionsInfo)
      compute_VP(intersectionsInfo)
      val = cv2.waitKey()
      if val == 27:
         break


   cv2.destroyWindow("Lineas")
