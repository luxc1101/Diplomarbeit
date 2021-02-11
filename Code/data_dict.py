import pathlib
import os
import pandas as pd

class data_dictionary():
        '''
        --------------
        DESCRIPTION:
        --------------
        this class is ueed to creat path and dir to save data

        --------------
        POPULAR MEMBERS:
        --------------
        creat_dir: to creat dirs to save data or pic

        --------------
        ATTRUBUTION
        --------------
        sv: data symbol
        rpath: root path
        ext: the data file with which ext will be read 
        filename: the name of file
        title: the data title
        title_dic: the dict zip filename und title

        '''

        def __init__(self, sv:str, rpath:str, ext:str):

                self.sv    = sv
                
                self.rpath = rpath
                
                self.ext   = ext
                
                self.spath = os.path.join(pathlib.PureWindowsPath(rpath).parents[0],'Output_{}'.format(sv))
                
                dirs  = [d for d in os.listdir(path = rpath) if os.path.isdir(os.path.join(rpath,d))]

                if len(dirs)!=0:
                        self.dirs = dirs
                else:
                        self.dirs = ['']

                self.point_num_all   = []
                
                self.point_num_train = []

                self.point_num_test  = []

                self.filenames = (
                        '{}_d'.format(sv),              # 0
                        '{}_I'.format(sv),              # 1
                        '{}_U'.format(sv),              # 2
                        '{}_R'.format(sv),              # 3
                        '{}_P'.format(sv),              # 4
                        '{}_OY'.format(sv),             # 5
                        '{}_UY'.format(sv),             # 6
                        '{}_F'.format(sv),              # 7
                        '{}_valley_id'.format(sv),      # 8
                        '{}_PQ'.format(sv),             # 9
                        # ----------------------------------
                        '{}_GQ'.format(sv),             # 10
                        '{}_MOW'.format(sv),            # 11
                        '{}_dt'.format(sv),             # 12
                        '{}_Qdot'.format(sv),           # 13
                        '{}_Qsgf'.format(sv),           # 14
                        '{}_Ai'.format(sv),             # 15
                        '{}_Pr'.format(sv),             # 16
                        '{}_dR'.format(sv),             # 17
                        '{}_R10'.format(sv),            # 18
                        '{}_EL'.format(sv),             # 19
                        '{}_Pv'.format(sv),             # 20
                        '{}_dP'.format(sv),             # 21
                        '{}_Pmax'.format(sv),           # 22
                        '{}_ETp'.format(sv),            # 23
                        '{}_ETS'.format(sv),            # 24
                        '{}_MET'.format(sv),            # 25
                        '{}_ser'.format(sv),            # 26
                        '{}_zone'.format(sv),           # 27
                        '{}_R90'.format(sv),            # 28
                        '{}_Isoll'.format(sv),          # 29
                        '{}_Fs1'.format(sv),            # 30
                        '{}_Fs2'.format(sv),            # 31
                        '{}_dF'.format(sv),             # 32
                        '{}_dI'.format(sv),             # 33
                        )

                self.title = (
                        'DURCHMESSER',                  # 0
                        'SCHWEISSSTROM',                # 1
                        'SCHWEISSSPANNUNG',             # 2
                        'PROZESSWIDERSTAND',            # 3
                        'PROZESSLEISTUNG',              # 4
                        'WEGMESSUNG OBEN',              # 5
                        'WEGMESSUNG UNTEN',             # 6
                        'ELEKTRODENKRAFT',              # 7
                        'VALLEYID',                     # 8
                        'PARTIELLENWAERMEMENGE',        # 9
                        # ----------------------------------
                        'GES.WAERMEMENGE',
                        'MEANOBEREWEGMESSUNG',
                        'ZEITDIFFERENZ',
                        'ERWAERMUNGSGESCHWINDIGKEIT',
                        'WAEMEBEDEUTUNG',
                        'ELEKTRODENFLAECHE',
                        'DRUCK',
                        'WIDERSTANDDIFFERENZ',
                        'WIDERSTAND10',
                        'LAEGENAENDERUNG', 
                        'LEISTUNGSABFALLGESCHWIN', 
                        'LEISTUNGSSCHWANKUNG',
                        'MAXLEISTUNG',
                        'EINDRUCKSTIEFPROZESSENDE', 
                        'EINDRUCKSTIEFSCHWEISSENDE', 
                        'MAXEINDRUCKSTIEF',
                        'SPEZIWIDERSTAND',
                        'ZONE',
                        'WIDERSTAND90',
                        'SOLLSTROM',
                        'KRAFTGRADIENT1',
                        'KRAFTGRADIENT2',
                        'KRAFTAENDERUNG',
                        'STROMAENDERUNG',
                        )

                self.Unit = (
                        ['I', '[kA]'],
                        ['U','[v]'],
                        ['R','[mohm]'],
                        ['P','[kW]'],
                        ['OY','[mm]'],
                        ['UY','[mm]'],
                        ['Fe','[kN]'],
                        ['Valley_id','[-]'],
                        ['Qpar','[J]'],
                        # ----------------------------------
                        ['Qges','[kJ]'],
                        ['MOW','[mm]'],
                        ['Delta_t%50','[s]'],
                        ['Qpdot', '[kJ/s]'],
                        ['Qsfg','J'],
                        ['Ai','[mm^2]'],
                        ['Press', '[N/mm^2]'],
                        ['Delta_R','[mohm]'],
                        ['R10','[mohm]'],
                        ['EL', '[mm]'],
                        ['Pv', '[kw/s]'],
                        ['delta_P', '[kW]'],
                        ['Pmax', '[kW]'],
                        ['ep', '[mm]'],
                        ['es', '[mm]'],
                        ['max_e', '[mm]'],
                        ['rho', '[mohm*mm^2/mm]'],
                        ['Z', '[-]'], 
                        ['R90','[mohm]'],
                        ['I_soll','[kA]'],
                        ['s1','[kN/s]'],
                        ['s2','[kN/s]'],
                        ['Delta_F','[kN]'],
                        ['Delta_I','[kA]'],
                        )

                self.title_unit_dic = dict(zip(self.title[1:], self.Unit))
                
                self.title_dic      = dict(zip(self.filenames, self.title))
                
                self.df_feature     = pd.DataFrame()

        def creat_dir(self, creat:bool):

                '''
                --------------
                PARAMETERS:
                --------------
                creat: bool  to create new folders or not

                --------------
                ATTRUBUTION:
                --------------
                csv_dir_path:   csv data dir path
                pic_dir_path:   picture dir path
                eff_dir_path:   effective dir path
                train_dir_path: trainig data dir path
                test_dir_path:  test data dir path
                all_dir_path:   all data dir path
                '''

                self.csv_dir_path   = os.path.join(self.spath,self.sv + '_CSV')
                self.pic_dir_path   = os.path.join(self.spath,self.sv + '_PIC')
                self.eff_dir_path   = os.path.join(self.spath,self.sv + '_EFF')
                self.train_dir_path = os.path.join(self.spath,self.sv + '_TRAIN', '')
                self.test_dir_path  = os.path.join(self.spath,self.sv + '_TEST' , '')
                self.all_dir_path   = os.path.join(self.spath,self.sv + '_ALL'  , '')
                print(self.spath)
                
                if creat:
                        os.makedirs(self.spath, exist_ok         = True)
                        os.makedirs(self.csv_dir_path,exist_ok   = True)
                        os.makedirs(self.pic_dir_path,exist_ok   = True)
                        os.makedirs(self.eff_dir_path,exist_ok   = True)
                        os.makedirs(self.train_dir_path,exist_ok = True)
                        os.makedirs(self.test_dir_path,exist_ok  = True)
                        os.makedirs(self.all_dir_path,exist_ok   = True)

                else:
                        print('no dirs be created')

        def data_save_dict(self):

                filenames_list_dict = {}
                
                for fi_ in self.filenames:
                        filenames_list_dict.setdefault(fi_,[])

                return filenames_list_dict
