import os
import pandas as pd
from datetime import date

class data_pre:
	'''
	-------------------------------------
	DESCRIPTION:
	-------------------------------------
	save the roh data from labview to csv

	12 kanal:
		- 0 Zeit
		- 1 Durchfluss
		- 2 Temperatur Vorlauf
		- 3 Temperatur Rücklauf
		- 4 Schweißspannung
		- 5 Elektrodenspannung
		- 6 Messstrom
		- 7 Elektrodenkraft
		- 8 Schweißstrom
		- 9 Referenzsignal
		- 10 Wegmessung Oben
		- 11 Wegmessung Unten
	-------------------------------------
	PARAMETER:
	-------------------------------------
	r_path: root path of labview data
	s_path: a new path to save the output data
	data_id: 0,1,2...(class variable)
	data_ext: ['.dat','.dat_h',]
	kanal: list which kanal be selected 
	save: bool save the data to csv datei or not
	verbose: bool show info or not
	-------------------------------------
	POPULAR MEMBERS:
	-------------------------------------
	data_to_df: methode
	write_csv: staticmethod

	-------------------------------------
	ATTRIBUTE:
	-------------------------------------
	data:		data Dataframe
	data_head:	labview .dat_h datei
	data_col:	columns info
	begin_id:	Iw1_beginn [index]
	ende_id:	Iw1_ende [index]
	samplerate:	sampling rate [Hz pro Kanal]
	thickness:	Blechdicke t1 [mm] + Blechdicke t2 [mm] 
	weildingtime:	Schwei遺eit [ms]
	electrodeforce:	Elektrodenkraft [kN]
	weldingcurrent:	Schweißstrom [kA]
	pointnumber:	point id
	sheet1:	material of 1. sheet
	sheet2: material of 2. sheet
	glue: klebstoff
	dp1: minimum diameters [mm] if the FE1_dp1 exists
	dp2: maximum diameters [mm] if the FE1_dp2 exists
	dw:	weld diameter [mm] (dp1 + dp2)/2
	-------------------------------------
	'''
	data_id = 0

	def __init__(self, r_path, s_path, data_ext = ['.dat','.dat_h',], kanal =[] ,save = True, verbose = True):
		
		self.r_path   = r_path
		
		self.s_path   = s_path
		
		self.data_ext = data_ext
		
		self.kanal    = kanal
		
		self.save     = save
		
		self.verbose  = verbose
	
	def open_file(self,path):
		
		with open(path) as f:
		
			lines = f.readlines()
		
		L = [ele.rstrip().split('\t') for ele in lines]
		
		return L
		
	def data_to_df(self):

		os.makedirs(self.s_path, exist_ok=True)
		
		dirs         = os.listdir(path = self.r_path)
		
		data_files   = [f for f in dirs if f.endswith(self.data_ext[0])]
		
		data_h_files = [f for f in dirs if f.endswith(self.data_ext[1])]
		
		data_dict    = dict(zip(data_h_files,data_files))
		
		data_h_path  = os.path.join(self.r_path,data_h_files[self.data_id])
		
		data_path    = os.path.join(self.r_path,data_dict[data_h_files[self.data_id]])
		
		if self.verbose:

			print('{} files in total'.format(len(data_dict)))
			
			print('{}  {}'.format(data_h_files[self.data_id],data_dict[data_h_files[self.data_id]]))
		
		# data head info
		L_h                 = self.open_file(data_h_path)
		
		self.data_name      = data_h_files[self.data_id].split('.')[0]
		
		self.data_head      = pd.DataFrame(L_h, columns = ['Exp_Info', 'Data']).set_index(keys = 'Exp_Info')
		
		self.begin_id       = int(float(self.data_head.loc['Iw1_beginn [index]'].apply(lambda x: x.replace(',','.'))))
		
		self.ende_id        = int(float(self.data_head.loc['Iw1_ende [index]'].apply(lambda x: x.replace(',','.'))))
		
		self.samplerate     = int(float(self.data_head.loc['Goldammer Samplerate [Hz pro Kanal]'].apply(lambda x: x.replace(',','.'))))
		
		self.thickness      = float(self.data_head.loc['Blechdicke t1 [mm]'].apply(lambda x: x.replace(',','.')))+float(self.data_head.loc['Blechdicke t2 [mm]'].apply(lambda x: x.replace(',','.')))
		
		self.weildingtime   = int(float(self.data_head.loc['Schwei遺eit [ms]'].apply(lambda x: x.replace(',','.'))))
		
		self.electrodeforce = float(self.data_head.loc['Elektrodenkraft [kN]'].apply(lambda x: x.replace(',','.')))
		
		self.weldingcurrent = float(self.data_head.loc['Schwei遱trom [kA]'].apply(lambda x: x.replace(',','.')))
		
		self.pointnumber    = int(self.data_head.loc['Punktnummer'].apply(lambda x: x.replace(',','.')))
		
		self.sheet1         =  self.data_head.loc['Blech 1 [t1]'][0]
		
		self.sheet2         =  self.data_head.loc['Blech 2 [t2]'][0]
		
		self.glue           = self.data_head.loc['Klebstoff'][0]
		
		idx                 = pd.Index(self.data_head.index)

		if 'FE1_dp1 [mm]' and 'FE1_dp2 [mm]' in idx: 

			FE1_dp1  = float(self.data_head.loc['FE1_dp1 [mm]'].apply(lambda x: x.replace(',','.')))
			
			FE1_dp2  = float(self.data_head.loc['FE1_dp2 [mm]'].apply(lambda x: x.replace(',','.')))
			
			self.dp1 = min(FE1_dp1, FE1_dp2)
			
			self.dp2 = max(FE1_dp1, FE1_dp2)
			
			self.dw  = round((self.dp1 + self.dp2)/2,3)

		if self.verbose:
			
			print('data head is done.')
		
		# data
		
		L_d                   = self.open_file(data_path)
		
		data                  = pd.DataFrame(L_d).replace(to_replace =[None], value= 'Unbenannt',)
		
		columns, unit, status = data.iloc[0,self.kanal], data.iloc[1,self.kanal], data.iloc[3,self.kanal]
		
		self.data_col         = [l for l in zip(columns,unit, status)]
		
		self.data             = pd.DataFrame(data.iloc[4:,self.kanal]).apply(lambda x: x.str.replace(',','.')).apply(lambda x: pd.to_numeric(x,errors='ignore')).reset_index(drop=True)
		
		self.data.columns     = list(columns)
		
		if self.verbose:
			
			print('data is done.')
		
		# save data head and data to save path
		if self.save:
			
			self.data_head.to_csv("{}/{}_h.csv".format(self.s_path,data_h_files[self.data_id].split('.')[0]))
			
			self.data.to_csv("{}/{}.csv".format(self.s_path,data_dict[data_h_files[self.data_id]].split('.')[0]),index = None)
			
			if self.verbose:
				
				print('save is done.')

	@staticmethod
	def write_csv(r_path, file_name = 'unbenannt.dat', title = 'UNBENANNT', welding_condi = [],sheet_info = [], ):
		'''
		----------------
		PARAMETER
		----------------
		r_path:	root path of labview data
		file_name: name of the .dat datei 
		title: the title of data in the file
		welding_condi : a list [electrodeforce, weildingtime, weldingcurrent] 
		sheet_info: a list [material, thickness in mm, glue(ohne or mit)]
		-----------------
		RETURN
		----------------
		csv_write : a file with basic information to be written
		'''
		csv_write = open(r_path + file_name,'w')
		
		csv_write.write("{:*^70s}".format(title) + '\n')
		
		csv_write.write('*' + '\n')
		
		csv_write.write('{:<25}{}'.format('*Quelle:',r_path) + '\n')
		
		csv_write.write('{:<25}{:<15}{:<10}{:<10}'.format('*SCHWEISSBEDINGUNGEN:','{}kN'.format(welding_condi[0]),'{}ms'.format(welding_condi[1]),'{}kA'.format(welding_condi[2])) + '\n')
		
		csv_write.write('{:<25}{:<15}{:<10}{:<10}'.format('*SHEET:',sheet_info[0],'{}mm'.format(sheet_info[1]),sheet_info[2]) + '\n')
		
		csv_write.write('{:<25}{}'.format('*AUTOR:','Xiaochuan Lu') + '\n')

		csv_write.write('{:<25}{}'.format('*DATUM:',date.today()) + '\n')

		csv_write.write('*' + '\n')

		csv_write.write('*' * 70 + '\n' * 2)

		return csv_write
