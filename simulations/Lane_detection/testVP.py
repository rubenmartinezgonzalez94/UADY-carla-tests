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

class vanishingPoint:
   def __init__(self, x, y, linesIdx, lines):
      self.x, self.y = x, y
      self.linesIdex = linesIdx
      self.lines = lines
      self.w=[]

      #Calculamos los pesos de cada linea en función de su longitud
      for l in lines[:]:
         self.w.append(l[3])
      sw = sum(self.w)
      for i in range(len(self.w)):
         self.w[i] /= sw

      # Calculamos la matriz de productos externos ponderados.
      M=np.zeros((3,3))
      i=0
      for l in self.lines:
         M += self.w[i] * np.outer(l[:3],l[:3])
         i += 1

      # Definimos el Punto de Fuga como el eigenvector asociado al eigenvalor
      # menor de la matriz acumuladora de productos exteriores.
      [l,V]=np.linalg.eig(M)
      mn = min(l)
      idxMin = l.tolist().index(mn)
      V[:,idxMin] /= V[2,idxMin]
      self.x = V[0, idxMin]
      self.y = V[1, idxMin]

   def __repr__(self):
      s = "VP=(%06.3f, %06.3f)\n" % (int(self.x), int(self.y))
      i = 0
      for l in self.lines[:]:
         s += "w=%f, line: %fx+%fy+%f=0\n" % (self.w[i], l[0], l[1], l[2])
         i += 1
      return s
   def __str__(self):
      s = "VP=(%f, %06.3f)\n" % (int(self.x), int(self.y))
      i = 0
      for l in self.lines[:]:
         s += "w=%f, line: %fx+%fy+%f=0\n" % (self.w[i], l[0], l[1], l[2])
         i += 1
      return s


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
   linesInfo = np.zeros((len(lines), 9))
   if lines is not None:
      for i, line in enumerate(lines):
         x1, y1, x2, y2 = line[0]
         M = np.array([[x1, y1, 1], [x2, y2, 1]])
         linesInfo[i, :3] = null_space(M)[:, 0] #Line Equation.
         linesInfo[i, 3] = np.sqrt((x1 - x2)**2 + (y1 - y2)**2) # Line segment lenght.
         linesInfo[i, 4] = np.arctan2(y1- y2, x1 - x2) # Line segment angle.
         linesInfo[i, 5:] =x1, y1, x2, y2 # Coordinate that bound the line segment.
   # Sort the array by lines angle
   linesInfo=linesInfo[linesInfo[:,4].argsort()]
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

def getSimilarLines(LI, threshold = 0.5):
   LI.linesInfo
   n = LI.nLines
   simil = []
   for i in range(n-1):
      a = LI.linesInfo[i, :3]
      a = a / la.norm(a)
      denA = 1 / np.sqrt(LI.linesInfo[i,0]**2 + LI.linesInfo[i,1]**2)
      for j in range(i+1, n):
         b = LI.linesInfo[j, :3]
         b = b / la.norm(b)
         denB = 1 / np.sqrt(LI.linesInfo[j,0]**2 + LI.linesInfo[j,1]**2)
   # Compute the distance between the one line into two points layin on the
   # other line, and viceversa.
         dist1 = np.sqrt(np.dot(LI.linesInfo[i, :3], np.hstack([LI.linesInfo[j, 5:7], 1]))**2) * denA
         dist2 = np.sqrt(np.dot(LI.linesInfo[i, :3], np.hstack([LI.linesInfo[j,  7:], 1]))**2) * denA
         dist3 = np.sqrt(np.dot(LI.linesInfo[j, :3], np.hstack([LI.linesInfo[i, 5:7], 1]))**2) * denB
         dist4 = np.sqrt(np.dot(LI.linesInfo[j, :3], np.hstack([LI.linesInfo[i,  7:], 1]))**2) * denB     
   # Selects the maximum distance. If two lines are similiar this distance
   # should be a small number.
         dist = max([dist1, dist2, dist3, dist4])
   # If the maximum distance between two lines is small enough, the indexes
   # to those lines, together with the distance are added as a tuple to a list.
         if threshold == 0 or dist < threshold:
            simil.append((i,j,dist))
   # The list with similar paired lines is returned.
   return simil

def drawSimilarLines(image, LI, thr = 1):
   winName = "Similar Lines"
   cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
   flag = True
   idx = 0
   img = image.copy()
   while flag == True:
      S = getSimilarLines(LI, thr)
      for s in S:
        
         print ("s=", s)
         print ("Linea[%d] = " % s[0], LI.linesInfo[s[0],:])
         print ("Linea[%d] = " % s[1], LI.linesInfo[s[1],:])
         print("+"*80,"\n")
         drawWholeLine(img, LI.linesInfo[s[0],:], (255,0,128), 1)
         drawWholeLine(img, LI.linesInfo[s[1],:], (255,128,0), 1)
         drawLine(img, LI.linesInfo[s[0]], color=(64,0,192), width=1)
         drawLine(img, LI.linesInfo[s[1]], color=(64,192,0), width=1)
      cv2.imshow(winName, img)
      val = cv2.waitKey(0)
      if val == 27:
         flag=False
   cv2.destroyWindow(winName)


def drawLine(img, L, color=(0,0,0), width=1):
    x1, y1, x2, y2 = int(L[5]), int(L[6]), int(L[7]), int(L[8])
    cv2.line(img, (x1, y1), (x2, y2), color, width)

def drawWholeLine(img, L, color, width):
   r, c, _ = img.shape

   linEq = L[:3].copy()
   imgBounds = np.zeros((4,3))
   imgBounds[0, :] = np.array([1, 0,   0]) #Left
   imgBounds[1, :] = np.array([1, 0, 1-c]) #Right
   imgBounds[2, :] = np.array([0, 1,   0]) #Top
   imgBounds[3, :] = np.array([0, 1, 1-r]) #Bottom

   intrSects = np.zeros((4,3))
   inBounds = [False] * 4
   for i in range(4):
      intrSects[i, :] = np.cross(linEq, imgBounds[i, :])
      intrSects[i, :] = intrSects[i, :] / intrSects[i, 2]
      if intrSects[i, 0] >= 0 and intrSects[i, 0] < c and intrSects[i, 1] >= 0 and intrSects[i, 1] < r:
         inBounds[i] = True
         
   X=[]
   Y=[]
   for i in range(4):
      if inBounds[i]:
         x, y = intrSects[i, :2]
         X.append(int(x))
         Y.append(int(y))
   
   cv2.line(img, (X[0], Y[0]), (X[1], Y[1]), color, width) 

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
      drawWholeLine(img, intInfo.linesInfo[i], (255,0,255), 1)

      drawWholeLine(img, intInfo.linesInfo[j], (255,0,255), 1)


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

def exploreVP(image, winName, VP):
   flag = True
   idx = 0
   while flag == True:
      img = image.copy()

      for v in VP:
         for L in v.lines:
            drawWholeLine(img, L, (0, 255, 0), 1)
         cv2.circle(img, (int(v.x), int(v.y)), radius=3 , color=(255, 0, 255), thickness=-1)
            
      cv2.imshow(winName, img)
      val = cv2.waitKey(0)
      if val == 27:
         flag = False
      
def keyTupleCluster(c):
   return c[1]

def compute_VP(inter, dThresh=1, minSize=2, weightLines = True):

   # Construimos los clusters de intersecciones similares
   data=inter.intersections[:,:2]

   AG = AgglomerativeClustering(distance_threshold = dThresh, n_clusters=None, compute_full_tree=True).fit(data)
   nLabels = max(AG.labels_)
   print ("leaves       = ", AG.n_leaves_)
   print ("num Features = ", AG.n_features_in_) 
   print("labels        = ", nLabels)
  

   # Obtenemos las listas de intersecciones en cáda cúmulo encontrado y
   # ordenamos dichas listas en terminos de su tamaño, las más grandes primero.
   clusters=[]
   for i in range(nLabels):
      leaves = [j for j in range(nLabels) if AG.labels_[j]==i]
      clusters.append(leaves)
   clustersSize = [(j,len(clusters[j])) for j in range(len(clusters))]
   clustersSize.sort(key=keyTupleCluster, reverse=True)
   
   
   #Extraemos los indices que cluster cuyo tamaño sea mayor que minSize
   #potVPIndex -> Indice de Vanishing Points Potenciales.
   potVPIndex=[clustersSize[k][0] for k in range(len(clustersSize)) if clustersSize[k][1] >= minSize]
   
   VP=[]
   for i in potVPIndex:

      # Para Cada cumulo, extraemos los índices de las lineas involucradas
      # en las intersecciones.
      linesIdx = set()
      for j in clusters[i]:
         for k in inter.intLinesIdx[j].tolist():
            linesIdx.add(k)
      nLines = len(linesIdx)

      vp = vanishingPoint(None, None, list(linesIdx), inter.linesInfo[list(linesIdx),:])
   
      VP.append(vp)

   return VP   
   
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
    #  drawSimilarLines(image, intersectionsInfo, 1)

      print ("lines found: ", intersectionsInfo.nLines)
      print ("intersections found: ", intersectionsInfo.nIntersections)
      print ("intersections Infinity found: ", intersectionsInfo.nIntersectionsInf)
      print ("-"*80, "\n")

     # exploreIntersections(image, "Lineas", intersectionsInfo)
      
      VP = compute_VP(intersectionsInfo,dThresh=0.5,minSize=3)
      print("VP's = ", VP)

      exploreVP (image, "Lineas", VP)

      val = cv2.waitKey()
      if val == 27:
         break


   cv2.destroyWindow("Lineas")
