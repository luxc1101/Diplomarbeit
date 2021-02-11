# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from nptdms import TdmsFile
from datetime import date
import pathlib
import re
import natsort
############################################################################################################
def data_write(
    data,
    w_path: str,
    title_dic: dict,
    filename: str,
    filenames_list_dic: dict,
    point_num: list,
    add_time: bool,
    title_unit_dic: dict,
    time = None,
    *args,
    **kwargs,
    ):

    '''
    this def must with class data_pre together use
    '''

    # select data according to key
    title = title_dic[filename]
    
    try:
        filenames_list_dic = filenames_list_dic[filename]
        if not filenames_list_dic:
            return
        # if len(filenames_list_dic)!=0:
        #     pass
        else:
            pass
    except:
        print(f'{title} is not exist!')
        return 

    csv_write = data_pre.write_csv(
        w_path        = w_path, 
        file_name     = '{}.dat'.format(filename),
        title         = title, 
        welding_condi = [data.electrodeforce, data.weldingtime, data.weldingcurrent],
        sheet_info    = [data.sheet1, data.sheet2, data.thickness]
        )

    if title not in [v for k, v in title_dic.items()][10:]:
    # write the time-related data
        if add_time:

            filenames_list_dic.insert(0,list(time))
            csv_write.write('{}'.format('time') + '\t')
        
        for p_num in point_num:
            csv_write.write('{}'.format(p_num) + '\t')
            if p_num == point_num[-1]:
                csv_write.write('\n')

        col_format = ''
        for _ in range(len(filenames_list_dic)):
            col_format += '{:.6f}\t'

        for x in zip(*filenames_list_dic):    
            csv_write.write(col_format.format(*x) + '\n')

    else:

        csv_write.write('Punktnummer' + '\t' + title_unit_dic[title][0] + '\n' 
                        + ' ' + '\t' + title_unit_dic[title][1] + '\n')
        
        for p_num,value in  zip(point_num,filenames_list_dic):
        
            csv_write.write(
                r'{:4d}'.format(p_num) + '\t'  
                + '{:.5f}'.format(value) + '\n'
                )
    print(f'{title} is done.')

############################################################################################################
def position_classification(number, row, col):
    '''
    position classification

    Edge distance of at least 10 mm, point distance of at least 30 mm.
    192 points per sheet in 12 rows 16 points each.
    Test piece at least 470 mm long and 350 mm wide.
    The 192 welds are followed by 8 test welds on an extra test piece made of the same material (30 mm x 250 mm), 
    with the same spacing and the same weld point division.

    weldingplan:

     https://i.loli.net/2020/10/30/7SJjyQnMZhciNAv.png

     https://i.loli.net/2020/10/30/1Kdl8NRYwZqAmgv.png
    '''
    pkt_l   = [pkt for pkt in range(number)]
    pkt_arr = np.array(pkt_l).reshape(row,col)
    for row in range(12):
        if row % 2 != 0:
            pkt_arr[row] = np.flipud(pkt_arr[row])

    zone_sel1 = list(pkt_arr[:,3:-3].reshape(-1))

    zone1     = [z1 for z1 in list(pkt_arr.reshape(-1)) if z1 not in zone_sel1]
    zone3     = list(pkt_arr[:,6:10].reshape(-1))
    zone2     = [z2 for z2 in list(pkt_arr.reshape(-1)) if z2 not in zone1 + zone3]

    return zone1, zone2, zone3

zone1, zone2, zone3 = position_classification(number = 192, row = 12, col = 16)

############################################################################################################

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
    dir_path: root path of labview data
    csv_s_path: where to save the output data
    pic_s_path: where to save the output pic
    eff_s_path: where to save the eff data
    file_id: 0,1,2...(class variable)
    data_ext: ['.tdms']
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
    data:       data Dataframe
    data_head:  .dat_h datei
    data_col:   columns info
    begin_id:   Iw1_beginn [index]
    ende_id:    Iw1_ende [index]
    samplerate: sampling rate [Hz pro Kanal]
    thickness:  Blechdicke t1 [mm] + Blechdicke t2 [mm] 
    weldingtime:   Schweisseit [ms]
    electrodeforce: Elektrodenkraft [kN]
    weldingcurrent: Schweissstrom [kA]
    pointnumber:    point id
    sheet1: material of 1. sheet
    sheet2: material of 2. sheet
    glue: klebstoff
    dp1: minimum diameters [mm] if the FE1_dp1 exists
    dp2: maximum diameters [mm] if the FE1_dp2 exists
    dw: weld diameter [mm] (dp1 + dp2)/2
    -------------------------------------
    '''
    file_id = 0

    def __init__(self, dir_path, csv_s_path, pic_s_path, eff_s_path ,data_ext: list,verbose: bool, save: bool):
        
        self.dir_path   = dir_path # the last dir path
        self.csv_s_path = csv_s_path # .csv save path
        self.pic_s_path = pic_s_path # pic save path
        self.eff_s_path = eff_s_path # eff save path
        self.data_ext   = data_ext # data extend
        self.verbose    = verbose
        self.save       = save

        if 'PB' not in pathlib.PurePosixPath(self.dir_path).name:
            if self.file_id in zone1:
                self.zone = 1
            elif self.file_id in zone2:
                self.zone = 2
            else:
                self.zone = 3
        else:
            self.zone = self.file_id # prüfblech positon can identify by file id

    def open_file(self,path):
        
        with open(path) as f:
        
            lines = f.readlines()
        
        L = [ele.rstrip().split('\t') for ele in lines]
        
        return L


    def id_correct(self, df, axis, basis_condi, other_condi: list, corr_id: list):
        '''
        Weil es immer Anzeigefehler für manche deu Wort mit 'ß','ö', usw gibt, Beim Lesen der Value nach Index, die mit 'ß' sind, 
        kann es immer Fehler auftreten. Deswegen durch dieses Method kann nach Bedingugen die Wörte korregiert werden.
        Because there are always display errors for some deu words with 'ß', 'ö', etc., errors can always occur when reading the value by index that are with 'ß'. 
        Therefore, through this method, the words can be corrected according to conditions.
        ------------------------
        df: the data
        axis: 0 index or 1 columns
        basis_condi: str, basic condition
        other_condi: list, if none can []
        corr_id: list, a list to instead of the selected false ID

        '''
        
        OG_id_l = []

        if axis == 0:

            df_id = df.index

        if axis == 1:

            df_id = df.columns 

        for idx in df_id:
            if len(other_condi):
                mlist = [True if oc in idx else False for oc in other_condi]
            else:
                mlist = [True]
            if basis_condi in idx and any(mlist):
                id_loc = df_id.get_loc(idx)
                OG_id  = df_id[id_loc]
                OG_id_l.append(OG_id)
        
        if len(OG_id_l) != len(corr_id):
            print('The list length must be the same. Maybe change same conditions')
        else:
            OG_id_corr_id_dic = dict(zip(OG_id_l,corr_id))

            if axis == 1:
                df.rename(columns=OG_id_corr_id_dic, inplace = True)
            if axis == 0:
                df.rename(index = OG_id_corr_id_dic, inplace = True)

        return df


    def data_to_df(self):

        try:
            os.makedirs(self.csv_s_path, exist_ok=True)
        except:
            print('No Path to create a dir to save csv.')
        try:
            os.makedirs(self.pic_s_path, exist_ok=True)
        except:
            print('No Path to create a dir to save pic.')
        try:
            os.makedirs(self.eff_s_path, exist_ok=True)
        except:
            print('No Path to create a dir to save eff.')

        unsorted_files   = [f  for f in os.listdir(path = self.dir_path) if f.endswith(self.data_ext[0]) ]

        data_files     =  natsort.natsorted(unsorted_files, key = lambda x: re.split('-P|{}'.format(self.data_ext[0]),x)[-2]) 

        self.file_name = '.'.join(data_files[self.file_id].split('.')[:-1]).replace(',','.')

        if '.tdms' in self.data_ext:

            # --------------------------------------------------------------------------------------------------------
            # read .tdms data and convert it to .csv data

            file_path        = os.path.join(self.dir_path,data_files[self.file_id])
            
            tdms_file        = TdmsFile.read(file_path)
            
            group            = tdms_file.groups()[0]
            
            group_name       = group.name 
            
            group_df         = group.as_dataframe(time_index=True).rename_axis('Zeit')
            
            channel_name_l   = [channel.name for channel in group.channels()]
            
            group_df.columns = channel_name_l
            
            data             = group_df.reset_index()

            # --------------------------------------------------------------------------------------------------------
            # read prop from .tdms datei

            group_head          = group.properties.items()
            
            group_head_df       = pd.DataFrame(group_head, columns =['Prop','Value']).set_index('Prop')

        else:

            unsorted_h_files = [f for f in os.listdir(path = self.dir_path) if f.endswith(self.data_ext[1])]

            data_h_files = natsort.natsorted(unsorted_h_files, key = lambda x: re.split('-P|{}'.format(self.data_ext[1]),x)[-2])

            data_dict    = dict(zip(data_h_files,data_files))

            data_h_path  = os.path.join(self.dir_path,data_h_files[self.file_id])

            data_path    = os.path.join(self.dir_path,data_dict[data_h_files[self.file_id]])

            # --------------------------------------------------------------------------------------------------------
            # read dat data and convert it to .csv data
            
            L_d  = self.open_file(data_path)
            
            data = pd.DataFrame(L_d)

            data = data.rename(columns=data.iloc[0]).iloc[4:,:12]
            
            data = pd.DataFrame(data).apply(lambda x: x.str.replace(',','.')).apply(lambda x: pd.to_numeric(x,errors='ignore')).reset_index(drop=True)
            
            # --------------------------------------------------------------------------------------------------------
            # read prop from dat_h datei

            L_h = self.open_file(data_h_path)

            group_head_df = pd.DataFrame(L_h, columns =['Prop','Value']).set_index('Prop')
        
        # because of 'ß' will show falsh index, therefore same index name must be changed ,whitch has 'ß' symbol
        
        
        group_head_df = self.id_correct(
            df          = group_head_df,
            axis        = 0,
            basis_condi = 'Schwei', 
            other_condi = ['[ms]','[kA]'],
            corr_id     = ['Schweisszeit [ms]','Schweissstrom [kA]']
            )

        self.data = self.id_correct(
            df = data,
            axis = 1,
            basis_condi = 'Schwei',
            other_condi = ['annung','rom'],
            corr_id = ['Schweissspannung','Schweissstrom']
            )

        self.samplerate     = int(float(group_head_df.loc['Goldammer Samplerate [Hz pro Kanal]'].apply(lambda x: x.replace(',','.'))))
        
        self.thickness      = float(group_head_df.loc['Blechdicke t1 [mm]'].apply(lambda x: x.replace(',','.')))+float(group_head_df.loc['Blechdicke t2 [mm]'].apply(lambda x: x.replace(',','.')))
        
        self.sheet1         =  group_head_df.loc['Blech 1 [t1]'].values[0]
        
        self.sheet2         =  group_head_df.loc['Blech 2 [t2]'].values[0]

        # self.sheet1         =  'HX340+Z100MB'

        # self.sheet2         =  'HX340+Z100MB'

        # all point number in this dir 2193-2200_PB0012 have falsh Punktnummer Prop, so here must be corrected

        try:

            if pathlib.PurePosixPath(self.dir_path).name in ['2001-2192_VB0011' ,'2193-2200_PB0012']:
            
                self.pointnumber = int(group_head_df.loc['Punktnummer'].apply(lambda x: x.replace(',','.'))) + 800

            # elif pathlib.PurePosixPath(self.dir_path).name in ['3193-3200_PB0017']:

                # self.pointnumber = int(re.split('P',self.file_name)[1])

            else:

                self.pointnumber = int(group_head_df.loc['Punktnummer'].apply(lambda x: x.replace(',','.')))
        except:

            self.pointnumber = int(re.split('P',self.file_name)[1])

        self.electrodeforce = float(group_head_df.loc['Elektrodenkraft [kN]'].apply(lambda x: x.replace(',','.')))
        
        self.weldingtime    = int(float(group_head_df.loc['Schweisszeit [ms]'].apply(lambda x: x.replace(',','.'))))
        
        self.weldingcurrent = float(group_head_df.loc['Schweissstrom [kA]'].apply(lambda x: x.replace(',','.')))
        
        self.squeezetime    = int(float(group_head_df.loc['Vorhaltezeit [ms]'].apply(lambda x: x.replace(',','.'))))
        
        self.holdtime       = int(float(group_head_df.loc['Nachhaltezeit [ms]'].apply(lambda x: x.replace(',','.'))))
        
        self.stiffness      = 2.345 # kN/mm

        idx = pd.Index(group_head_df.index)

        if 'FE1_dp1 [mm]' and 'FE1_dp2 [mm]' in idx: 
            
            FE1_dp1  = float(group_head_df.loc['FE1_dp1 [mm]'].apply(lambda x: x.replace(',','.')))
            
            FE1_dp2  = float(group_head_df.loc['FE1_dp2 [mm]'].apply(lambda x: x.replace(',','.')))
            
            self.dp1 = min(FE1_dp1, FE1_dp2)
            
            self.dp2 = max(FE1_dp1, FE1_dp2)
            
            self.dw  = round((self.dp1 + self.dp2)/2,3)

        self.data_head = group_head_df

        # print(self.data_head)

        if self.verbose:

            print('Conversion {} is done.'.format(self.file_name))

        if self.save:

            try:

                self.data.to_csv(os.path.join(self.csv_s_path,self.file_name + '.csv'),index = None)

                self.data_head.to_csv(os.path.join(self.csv_s_path,self.file_name + '.dat_h'),sep='\t')

                if self.verbose:

                    print('Data save is done.')
            except:

                if self.verbose:

                    print('Data file cannot be saved, path may be missing')


    @staticmethod
    def write_csv(w_path, file_name = 'unbenannt.dat', title = 'UNBENANNT', welding_condi = [],sheet_info = [], ):
        '''
        ----------------
        PARAMETER
        ----------------
        w_path: root path of labview data
        file_name: name of the .dat datei 
        title: the title of data in the file
        welding_condi : a list [electrodeforce, weldingtime, weldingcurrent] 
        sheet_info: a list [material, thickness in mm, glue(ohne or mit)]
        -----------------
        RETURN
        ----------------
        csv_write : a file with basic information to be written
        '''
        csv_write = open(w_path + file_name,'w')
        
        csv_write.write("{:*^70s}".format(title) + '\n')
        
        csv_write.write('*' + '\n')
        
        csv_write.write('{:<25}{}'.format('*Quelle:',w_path) + '\n')
        
        csv_write.write('{:<25}{:<15}{:<10}{:<10}'.format('*SCHWEISSBEDINGUNGEN:','{}kN'.format(welding_condi[0]),'{}ms'.format(welding_condi[1]),'{}kA'.format(welding_condi[2])) + '\n')
        
        csv_write.write('{:<25}{:<15}{:<15}{:<10}'.format('*SHEET:',sheet_info[0], sheet_info[1],'{}mm'.format(sheet_info[2])) + '\n')
        
        csv_write.write('{:<25}{}'.format('*AUTOR:','Xiaochuan Lu') + '\n')

        csv_write.write('{:<25}{}'.format('*DATUM:',date.today()) + '\n')

        csv_write.write('*' + '\n')

        csv_write.write('*' * 70 + '\n' * 2)

        return csv_write
    







