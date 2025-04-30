#! /usr/bin/env python3
# coding: UTF-8

import numpy as np
import numpy.random as rnd
from matplotlib import pyplot as plt

class Ransac:
	def __init__(self, sigma = 1, s = 2, e = 0.5, m = 1):
		self.sigma = sigma       # Error Standard Deviation.
		
		self.s = s               # Cardinality of the minimal subset
		                         # needed to fit the data to the model.

		self.e = e               # Outlier proportion.

		self.nSamples = None     # Number of samples in the data set.

		self.m = m               # Dimension. Needed to compute the Threshold.

		self.M = None            # Matrix that store the measurements
		                         # to be fit to the model.

		self.outliersIdx = None  # Outliers indices.
		self.inliersIdx  = None  # Inliers indices.
		self.inliersMask = None
		self.coefs = None        # Model coeficients.

		self.estimator = None

		self.threshTable ={1:3.84, 2:5.99, 3:7.81}

		if (self.estimator == None):
			self.estimator = estimatorLine2D()
		
    # Function that uses ransac algorithm to find the set of inliers.
	def fitRansac(self, M, thrFact=1.):
		# First we found the number of samples needed in order to ensure
		# that the probability p, that one random sample is free from
		# outilers is 0.99.

		# Get the data matrix size
		mR, mC = M.shape

		# Missing: throw a exception if m < self.s

		self.nSamples = 2 * int(np.ceil(np.log (1-.99)/np.log(1-np.power((1-self.e), self.s))))
	
		self.thresholdSq = thrFact**2 * self.threshTable[self.m] * np.power(self.sigma, 2)
		self.threshold = np.sqrt(self.thresholdSq)

		idxs =np.zeros((self.s),dtype="int64")
		
		T = int(np.ceil((1. - self.e) * mR))
		
		for i in range(self.nSamples):

			# Selects s distinct indices in the interval [0,m)
			if self.s < mR:
				idxs[0] = int(rnd.randint(0,mR))
				for j in range(1,self.s):
					value = int(rnd.randint(0,mR))
					while (value in idxs ):
						value = int(rnd.randint(0,mR))
					idxs[j] = value
			else:
				print ("fitRansac:I should throw an exception here", (self.s, mR))
				pass #I should throw an exception here.

			sol, success = self.estimator.fitMinimal(M, idxs)

			if success == False:
				continue
			# Determine which points in the dataset are inliers

			iNrm=1. / np.sqrt(sol[0] ** 2 + sol[1] ** 2)
			inliersMask = np.full((mR),False)
			inliersIdx = np.full((mR,), -1, dtype='int64')
			outliersIdx = np.full((mR,), -1, dtype='int64')
			nInliers = nOutliers = 0
			error = 0.
			for j in range(mR):
				distSq = self.estimator.minDistance(M[j,:])
				if distSq < self.threshold:
		   			inliersMask = True
		   			inliersIdx[nInliers] = j
		   			nInliers += 1
		   			error += distSq 
				else:
					outliersIdx[nOutliers] = j
					nOutliers += 1
			if nInliers != 0:
				error /= nInliers
			else:
				error = None

			inliersIdx = inliersIdx[:nInliers]
			outliersIdx = outliersIdx[:nOutliers]
			if i == 0:
				self.inliersMask = inliersMask
				self.inliersIdx = inliersIdx
				self.outliersIdx = outliersIdx
			elif len(inliersIdx) > len(self.inliersIdx):
				self.inliersMask = inliersMask
				self.inliersIdx = inliersIdx
				self.outliersIdx = outliersIdx


			if len(self.inliersIdx) > T:
				print("Abortando la misión: %d > %d" % (len(self.inliersIdx), T))
				break

		coefs = self.estimator.fitBest(M,self.inliersIdx)

		return coefs, error

		# Function that uses an adaptive ransac algorithm to find
		# the inliers set.
	def fitAdaptiveRansac(self):
		pass

class estimatorLine2D:
	def __init__(self):
		self.coefs = np.zeros(2)
		self.intercept = None
		self.iNormSq = None
		self.iNorm = None
	# Function that exactly fits the model to the data.
	def fitMinimal(self, M, idxs):
		
		idx0 = idxs[0]
		idx1 = idxs[1]
			
		normM = M[idx1,0] * M[idx0,1] - M[idx0,0] * M[idx1,1]
		if normM != 0:
			A =-(M[idx0,1] - M[idx1,1]) / normM
			B =(M[idx0,0] - M[idx1,0]) / normM
			self.coefs = np.array([A,B])
			self.intercept = 1.
			self.iNormSq = 1. / (A * A + B * B)
			self.iNorm = np.sqrt(self.iNormSq)
			return np.array([A, B, 1.]), True
		else:
			self.coefs = None
			self.intercept = None
			self.iNormSq = None
			self.iNorm = None
			return None, False
	
		
		
    # Function that find the model coefficients that best fit the inliers
    # stored in idxs.
	def fitBest(self, M, idxs=None):
		r, c = M.shape
		n = len(idxs)
		m = np.zeros((n, c))
		cv = -np.ones((n, 1))
		m = M[idxs,:]
		
		[self.coefs,_,_,_]=np.linalg.lstsq(m,cv,rcond=None)

		self.iNorm = 1. / np.linalg.norm(self.coefs)
		self.iNormSq = np.power(self.iNorm, 2)
		self.intercept = 1.

		return np.append(self.coefs,self.intercept)

	def predict(self, X, indIdx):
		c = len(self.coefs)
		if indIdx<0 or indIdx >= c:
			#I should throw and exception here.
			return
		P = -np.hstack([X[:,:indIdx],X[:,(indIdx+1):]])
		q = self.coefs[np.hstack([np.arange(indIdx), np.arange(indIdx + 1, c)])]
		y = (np.dot(P,q) - self.intercept) / self.coefs[indIdx]
		return y

	def Error(self, M, idxs):
		n = len(idxs)
		error = 0
		for i in range(n):
			error += self.iNormSq * np.power(self.coefs[0] * M[idxs[i],0] + self.coefs[1] * M[idxs[i],1] + self.intercept, 2)
		error /= n

	def minDistanceSq(self, x):
		return self.iNormSq * np.power(self.coefs[0] * x[0] + self.coefs[1] * x[1] + self.intercept, 2)

	def minDistance(self, x):
		return self.iNorm * np.abs(self.coefs[0] * x[0] + self.coefs[1] * x[1] + self.intercept)

