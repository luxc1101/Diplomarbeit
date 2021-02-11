import numpy as np
import pandas as pd
from scipy import signal
from kalmanfilter import KalmanFilter, KalmanFilter_simple
import sys
from tqdm.auto import tqdm

class effectiv_trans:
	'''
	------------------
	PARAMETER:
	------------------
	data: data.iloc[] current
	begin_id: int begin id
	end_id: int end id
	threshold: allow minimum current value
	order: how many points on each side to use for the comparison to consider
	col_name : header of the column
	------------------
	POPULAR MEMBERS:
	------------------
	ID_search:		function
	effectiv_wert:		function
	gradient_filter:	staticmethod
	butter_lowpass_filter:	staticmethod
	savitzky_golay_filter:	staticmethod
	------------------
	ATTRIBUTE:
	------------------
	valley_id: ndarray valley id
	preak_id: ndarray peak id	
	'''
	def __init__(self, data, begin_id, end_id, order = None ,threshold = None, col_name = None):
		
		self.data      = data
		
		self.begin_id  = begin_id
		
		self.end_id    = end_id
		
		self.threshold = threshold
		
		self.order     = order
		
		self.col_name  = col_name

	def forward_s(self):
	    
	    while True:
	    
	        if self.data[self.begin_id] < self.threshold:
	    
	            break
	    
	        if self.begin_id <0:
	    
	            print('No results found for new begin id.')
	    
	            break
	    
	        self.begin_id -= 1

	    self.begin_id = self.begin_id + 1


	def backward_s(self):
	    
	    while True:
	    
	        if self.data[self.end_id] < self.threshold:
	    
	            break
	    
	        if self.end_id >= len(self.data):
	    
	            print('No results found for new end id. ')
	    
	            break
	    
	        self.end_id += 1

	    self.end_id = self.end_id-1

	def ID_search(self, forward =1,backward = 1):
		'''
		----------------
		PARAMETER
		----------------
		forward: 1 or 0 activate forward search or not
		backward: 1 or 0 activate backward search or not
		-----------------
		RETURN
		----------------
		new begin id: int 
		new end id:  int
		'''
		if (forward == 1) and (backward == 0):
		
			self.forward_s()
		
			return self.begin_id, self.end_id

		if (forward == 0) and (backward == 1):
		
			self.backward_s()
		
			return self.begin_id, self.end_id
		
		if (forward == 1) and (backward == 1):
		
			self.forward_s()
		
			self.backward_s()
		
			return self.begin_id, self.end_id

		else:
		
			print('No search for new begin id and new end id, use default id')
		
			return self.begin_id, self.end_id

	def find_peak_id(self):
		'''
		------------------
		return:
		------------------
		peak id: ndarray
		'''
		data = self.data[self.begin_id:self.end_id]
		
		peak_id_less_equal = signal.argrelextrema(np.array(data), np.greater_equal, order = self.order)[0] + self.begin_id
		
		for i in peak_id_less_equal:
		
			if abs(np.subtract(data[i],data[i+1]))>1:
		
				data[i] = data[i+1]
		
				peak_id_less_equal = signal.argrelextrema(np.array(data), np.greater_equal, order = self.order)[0] + self.begin_id

		repeat_peak_id = np.array([peak_id_less_equal[i+1] for i in range(len(peak_id_less_equal)-1) 
		if (data[peak_id_less_equal[i]] == data[peak_id_less_equal[i+1]]) and (np.subtract(peak_id_less_equal[i+1],peak_id_less_equal[i])<self.order)])

		peak_id = np.setdiff1d(peak_id_less_equal,repeat_peak_id)
		
		return peak_id

	def find_valley_id(self):
		'''
		------------------
		return:
		------------------
		valley id: ndarray
		'''
		data = self.data[self.begin_id:self.end_id]	
		
		valley_id_less_equal = signal.argrelextrema(np.array(data), np.less_equal, order = self.order)[0] + self.begin_id
		
		for i in valley_id_less_equal[1:-1]:
		
			if abs(np.subtract(data[i],data[i+1])) > 1:
		
				data[i] = data[i+1]
		
				valley_id_less_equal = signal.argrelextrema(np.array(data), np.less_equal, order = self.order)[0] + self.begin_id

		repeat_valley_id = np.array([valley_id_less_equal[i+1] for i in range(len(valley_id_less_equal)-1) 
		if (data[valley_id_less_equal[i]] == data[valley_id_less_equal[i+1]]) and (np.subtract(valley_id_less_equal[i+1],valley_id_less_equal[i])<self.order)])
		
		valley_id = np.setdiff1d(valley_id_less_equal,repeat_valley_id)
		
		return valley_id

	def effectiv_wert(self, id_search = True, valley_id = None, peak_id = None, EGM_window_width = 3):
		'''
		----------------
		DESCRIPTION
		----------------
		zuerst wird eine ID-Recherche durchgeführt, um die Begin ID und End ID von Stromsdaten zu bestimmen. 
		Aber für Spannungsdaten ist die ID Recherche nicht mehr notwendig, weil die Begin ID und End ID von 
		Spannungsdaten genau so wie Stromsdaten sind, deswegen kann man einfach alle ermittelt ID aus Stromsdaten
		bei Ermittelung der effecktiven Werte von Spannungsdaten direkt einsetzen.
		um die Schwankung von ermittelten effecktiven Werte zu verringern, wird eine EGM Methode 
		(Einseitiger gleitender Mittelwert) benutzt: 
		
		https://www.youtube.com/watch?v=dUEogTFQ_HM&list=LL16Ud9cJ24XuCwG0jU__6sw&index=2&t=0s

		vorgegebene Fensterbreite ist 3, dh. einmal wird die Mittelwert von 3 Zyklen ermittelt.
		je breiter das Fenster ist, desto glatter ist die effective Kurve, aber die kann auch größere 
		Phasenverschiebung verursachen.

		Die ersten 4 Zyklen (rms1, reise_cycle = Anstieg Zyklen - 1) ist normaleweise für Anstieg der Strom von 0 bis Schweißstrom. 
		Wegen der heftigen Änderung des Stroms wird die ersten 4 Zyklen keine EGM Methode eingesetzt. 
		Die Situation ist ebenfall für den letzten Zyklus.

		Aber biite beachten die Mittelwerte für die letzte ein paar Zyklen, die reste Anzahl der Zyklen ist kleiter als
		Fensterbreite. Bei diesen Fall ist die Fensterbreite kann flexibel sein, wie in rms2
		----------------
		PARAMETER
		----------------
		id_search: bool 
		valley_id: ndarray valley id
		peak_id: ndarray peak id
		EGM_window_width: the width of Filter window basis an EGM methode
		-----------------
		RETURN
		----------------
		data pd Series
		'''
		if id_search:
		
			begin_id, end_id = self.ID_search(1,1)
		
			self.valley_id, self.peak_id = self.find_valley_id(), self.find_peak_id()
		
		else:
		
			begin_id, end_id = self.begin_id, self.end_id
		
			self.valley_id = valley_id

		data = self.data[begin_id:end_id]
		
		if self.valley_id is None:
		
			print('Error: Missing valley id.')
		
			return None
		
		else:
		
		    repeat_num = np.array([self.valley_id[i+1] - self.valley_id[i] for i in range(len(self.valley_id)-1)])

		    rise_cycle = 3

		    rms1 = np.array([np.sqrt(np.mean(data[start_id + np.arange(add_num)]**2)) 
		    	for start_id, add_num in zip(self.valley_id[:-1],repeat_num) 
		    	if np.where(self.valley_id[:-1] == start_id)[0][0]<=rise_cycle])

		    k = EGM_window_width - 2
		    rms2 = []
		    for start_id in self.valley_id[:-1]:
		    	i = np.where(self.valley_id[:-1] == start_id)[0][0]
		    	if i>=(len(self.valley_id[:-1])-EGM_window_width):
		    		add_num = repeat_num[i]
		    		if i < len(self.valley_id[:-3]):
		    			d = i
		    			for _ in range(k):
		    				d+=1
		    				add_num  += repeat_num[d]
		    			k-=1

		    		rms = np.sqrt(np.mean(data[start_id + np.arange(add_num)]**2))
		    		rms2.append(rms)
		    		
		    rms2 = np.array(rms2)	
	
		    # rms2 = np.array([np.sqrt(np.mean(data[start_id + np.arange(add_num)]**2)) 
		    # 	for start_id, add_num in zip(self.valley_id[:-1],repeat_num) 
		    # 	if np.where(self.valley_id[:-1] == start_id)[0][0]>=(len(self.valley_id[:-1])-window_width)])

		    rms3 = []
		    for start_id in self.valley_id[:-1]:	
		    	i = np.where(self.valley_id[:-1] == start_id)[0][0]
		    	if rise_cycle<i<(len(self.valley_id[:-1])-EGM_window_width):
		    		add_num = repeat_num[i]
		    		for _ in range(EGM_window_width-1):
		    			i +=1
		    			add_num += repeat_num[i]
	    			
		    		rms = np.sqrt(np.mean(data[start_id + np.arange(add_num)]**2))
		    		rms3.append(rms)
		    		
		    rms3 = np.array(rms3)

		    rms = np.concatenate((rms1,rms3,rms2))
		    rms_all = np.array([rp_value for rp_num, rp_value in zip(repeat_num, rms) for _ in range(rp_num)])
			

		    if len(data)!=len(rms_all):
		
		        for _ in range(abs(len(data)-len(rms_all))):
		
		            rms_all = np.append(rms_all,rms[-1])

		    roh_data = self.data.copy()

		    roh_data[0:begin_id], roh_data[begin_id:end_id], roh_data[end_id:] = 0, rms_all, 0
		
		    data_new = pd.Series(roh_data, index = roh_data.index, name = self.col_name)
		
		    return data_new
	
	@staticmethod
	def gradient_filter(data, col_num: int, threshold, verbose: bool):

		x = data.iloc[:,col_num]
		
		x_ = x.copy()
		
		x_dot = np.gradient(np.array(x_),edge_order=1)
		
		x_dot_abs = abs(x_dot)
		
		if max(x_dot_abs) > threshold:
		
			mask = x_dot_abs>threshold
		
			if verbose:
		
				if col_num == 10:
		
					print('extreme gradient {} in the way of the upper electrode'.format(x_dot_abs[mask]))
		
				else:
		
					print('extreme gradient {} in the way of the lower electrode'.format(x_dot_abs[mask]))
		
			id_l = np.array([j for i in x_dot_abs[mask] for j in np.where(x_dot_abs == i)[0]])
		
			id_l.sort()
		
			id_l = np.unique(id_l)
		
			id_l_new = []
		
			for q in range(len(id_l)-1):
		
				diff = id_l[q+1] - id_l[q]
		
				if diff == 2:
		
					id_l_new.extend([id_l[q],id_l[q+1]])
		
			if len(id_l_new):
		
				if len(id_l_new) % 2 == 0:
		
					i = 0
		
					targ_id, mean_value = [],[]
		
					for _ in range(int(len(id_l_new)/2)):
		
						targ_id.append(int((id_l_new[i+1]+id_l_new[i])/2))
		
						mean_value.append((x_[id_l_new[i+1]]+x_[id_l_new[i]])/2)
		
						i += 2
		
					for k,v in zip(targ_id,mean_value):
		
						x_[k] = v
		return x_

	@staticmethod
	def butter_lowpass_filter(data,col_num,cutoff,Fs,order):
		'''
		----------------
		PARAMETER
		----------------
		data: data.Dateframe 
		col_mun: columns number
		cutoff: cutoff frequency
		Fs: Sampling rate
		order : Polynomial order of the signal
		-----------------
		RETURN
		----------------
		y : Filtered signal  
		'''
		normal_cutoff = 2*cutoff/Fs
		
		b,a = signal.butter(order,normal_cutoff,'low',analog=False)
		
		y = signal.filtfilt(b,a,data.iloc[:,col_num])
		
		return y

	@staticmethod
	def savitzky_golay_filter(data, col_num, threshold, window_width = None, polyorder = 1, verbose = True):
		'''
		----------------
		DESCRIPTION
		----------------
		Das Savitzky-Golay-Filter ist ein mathematischer Glättungsfilter in der Signalverarbeitung
		Die durch Sensor gemessten Signal könnten machmal unplausibele Werte existieren. Man muss solche Werte korregieren oder ignorieren.
		Die unplausibelen Werte bedeutet, dass extrem Value in genzem Signal vorhanden sein kann. 
		Deshabel kann die Gradient um diesen Punkt herum extrem groß oder wenig sein.
		
		Zuerst: kann man die Gradient von ganzen Signal ermitteln, dann stellt ein Grenz zur Maske, die kann die extreme Value und ID makieren.
		Dann werden die extremen Value durch die Mittelwert von vorherige Wert und nächste Wert dieses Punktes ersetzt.

		Danach: kann man die Savitzky-Golay-Methode durchführen
		----------------
		PARAMETER
		----------------
		data: data.Dateframe 
		col_mun: columns number
		threshold: based on the gradient analysis give a limit to filtering extremely gradient
		window_width: the length of the filter window
		polyorder : The order of the polynomial used to fit the samples
		verbose: bool show the extreme grandient
		-----------------
		RETURN
		----------------
		x_ : Filtered signal  
		'''
		x_ = effectiv_trans.gradient_filter(data = data, col_num = col_num, threshold = threshold,verbose = verbose)
		
		x_ = x_.to_frame()

		x_ = effectiv_trans.butter_lowpass_filter(data = x_, col_num = 0,cutoff = 300, Fs = 100000, order = 3)

		window_width_l =[window_width]
		
		for _ in range(3):
			
			window_width *= 3 
			
			window_width_l.append(window_width)

		for ww in window_width_l:
		
			x_ = signal.savgol_filter(x_,ww,polyorder)

		return x_

	@staticmethod
	def kalman_filter(data, col_num, threshold, Q: float ,EGM: bool, EGM_window_width = 800, verbose = True):
		'''
		----------------
		DESCRIPTION
		----------------
		Simple implementation of a Kalman filter based on:
		http://www.cs.unc.edu/~welch/kalman/kalmanIntro.html
		'''
		Measurement = effectiv_trans.gradient_filter(data = data, col_num = col_num, threshold = threshold,verbose = verbose)	

		P  = np.diag(np.array(1.0).reshape(-1))
		F  = np.matrix(1.0)
		H  = F
		R  = np.matrix(0.1**2)
		Q  = Q
		G  = np.matrix(1.0)
		Q  = G * (G.T) * Q 
		Z  = np.matrix(Measurement[0])
		X  = Z
		kf = KalmanFilter(X, P, F, Q, Z, H, R)
		X_ = [X[0,0]]


		for i in tqdm(range(1, len(Measurement)),desc = "Kalman filter...",ascii=False, ncols=75):
			# Predict
			(X, P) = kf.predict(X, P, w = 0)
			# Update
			(X, P) = kf.update(X, P, Z)

			Z = np.matrix(Measurement[i])
			
			X_.append(X[0,0])


		if EGM:		

			EGM = [X_[0]]
		
			n = EGM_window_width
		
			for i in tqdm(range(1,len(Measurement)-n+1),desc = "EGM filter...",ascii=False, ncols=80):
				
				EGM.append(np.mean(X_[i:i+n]))

			EGM.extend(X_[len(Measurement)-n+1:])

			return EGM
		
		else:
			
			return X_

	@staticmethod
	def kalman_filter_simple(data, col_num, threshold, R, Q, EGM: bool, EGM_window_width = 800, verbose = True):
		'''
		----------------
		DESCRIPTION
		----------------
		Simple implementation of a Kalman filter based on:
		http://www.cs.unc.edu/~welch/kalman/kalmanIntro.html
		'''
		Measurement = effectiv_trans.gradient_filter(data = data, col_num = col_num, threshold = threshold,verbose = verbose)

		kf = KalmanFilter_simple(R = R, Q = Q, n_iter = len(Measurement), data = Measurement)

		X_, P = kf.KFS()

		if EGM:

			EGM = [X_[0]]

			n = EGM_window_width
		
			for i in range(1,len(Measurement)-n+1):
				
				EGM.append(np.mean(X_[i:i+n]))

			EGM.extend(X_[len(Measurement)-n+1:])

			return EGM

		else:

			return X_



