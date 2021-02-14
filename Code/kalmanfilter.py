from numpy.linalg import inv
from numpy import identity
import numpy as np
# import matplotlib.pyplot as plt

'''
Simple implementation of a Kalman filter based on:
http://www.cs.unc.edu/~welch/kalman/kalmanIntro.html
'''

class KalmanFilter:
	def __init__(self, X, P, F, Q, Z, H, R):
		"""
		Initialise the filter
		Args:
		    X: State estimate
		    P: Estimate covaConfigureriance
		    F: State transition model
		    Q: Process noise covariance
		    Z: Measurement of the state X
		    H: Observation model
		    R: Observation noise covariance
		"""
		self.X = X
		self.P = P
		self.F = F
		self.Q = Q
		self.Z = Z
		self.H = H
		self.R = R

	def predict(self, X, P, w =0):
		"""
		Predict the future state
		Args:
		    X: State estimate
		    P: Estimate covariance
		    w: Process noise
		Returns:
		    updated (X, P)
        """	
		X = self.F * X + w
		P = self.F * P * (self.F.T) + self.Q
        
		return(X, P)

	def update(self, X, P, Z):
		"""
		Update the Kalman Filter from a measurement
		Args:
		    X: State estimate
		    P: Estimate covariance
		    Z: State measurement
		Returns:
		    updated (X, P)
		"""
		K = P * (self.H.T) * inv(self.H * P * (self.H.T) + self.R)
		X += K * (Z - self.H * X)
		P = (identity(P.shape[1]) - K * self.H.T) * P

		return (X, P)


class KalmanFilter_simple:
	def __init__(self, R, Q, n_iter, data):
		
		self.R = R
		
		self.Q = Q
		
		self.n_iter = n_iter
		
		self.data = data

	def KFS(self):
		
		z = np.array(self.data)

		sz = (self.n_iter,) # size of array

		xhat = np.zeros(sz) # a posteri estimate of x

		P = np.zeros(sz) # a posteri error estimate

		xhatminus = np.zeros(sz) # a priori estimate of x
		
		Pminus = np.zeros(sz)    # a priori error estimate

		K = np.zeros(sz)

		xhat[0] = z[0]

		P[0] = 1

		for k in range(1, self.n_iter):
			# time update
			xhatminus[k] = xhat[k-1]
			Pminus[k] = P[k-1] + self.Q

			# measurement update
			K[k] = 	Pminus[k]/(Pminus[k] + self.R)
			xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])
			P[k] = (1-K[k]) * Pminus[k]


		return (xhat, P)

