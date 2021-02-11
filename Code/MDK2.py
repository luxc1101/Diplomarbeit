import numpy as np
import pandas as pd
import pathlib
import os
import matplotlib.pyplot as plt
import itertools
from matplotlib import cm
import concurrent.futures
from scipy import integrate
from scipy.interpolate import interp1d, griddata
from scipy.signal import savgol_filter
from scipy.fft import fft
from kneed import KneeLocator
from matplotlib import cm
import re
import natsort

from sklearn.metrics import mean_absolute_error, mean_squared_error

from labvtocsv2 import data_pre, data_write
from eff_trans2 import effectiv_trans, Q_modell, conta_area, R_modell, F_modell
from figsave import save_fig
from Fuzzy_logic import fuzzy_logic
import ruptures as rpt
from data_dict import data_dictionary
# ---------------------------------------------------------------------------------------------------------------
sv    = 'Schweissdata_IGF_plot'
# ext = ['.dat','.dat_h']
ext   = ['.tdms']
rpath = 'E:/Prozessdaten/MDK2/Schweissdata/Schweissdata/'
# rpath = 'E:/Prozessdaten/MDK1/Testdaten/Problem/'
# rpath = 'E:/Prozessdaten/MDK2/Testdaten/2020-03-16_MDK2-Spritzer/'
# rpath = 'E:/Prozessdaten/MDK1/Testdaten/2020-03-16_MDK1-Spritzer/'
# rpath = 'E:/Prozessdaten/MDK2/Testdaten/2020-12-07_B/'
# rpath = 'E:/Prozessdaten/MDK1/Testdaten/Problem/'

data_dic = data_dictionary(sv = sv, rpath = rpath, ext = ext)
data_dic.creat_dir(creat = True)

data_list_dic_all   = data_dic.data_save_dict()
data_list_dic_test  = data_dic.data_save_dict()
data_list_dic_train = data_dic.data_save_dict()

filenames = data_dic.filenames
title     = data_dic.title
title_dic = data_dic.title_dic
dirs      = data_dic.dirs
spath     = data_dic.spath


point_num_all   = data_dic.point_num_all
point_num_train = data_dic.point_num_train
point_num_test  = data_dic.point_num_test

title_unit = data_dic.title_unit_dic

EGM_fensterbreite = 6 # die beste Fensterbreite

# df_feature_all = pd.DataFrame(columns= [col.split('_')[-1] for col in filenames[9:]])
df_feature_all = data_dic.df_feature
df_feature_test = data_dic.df_feature
df_feature_train = data_dic.df_feature

def eff_write(time, I_rms, U_rms, R, P, OY, UY, Force):
    df_eff   = pd.DataFrame(time)
    eff_dict = dict(zip(data_list_dic_all.keys(),(I_rms,U_rms,R,P,OY,UY,Force)))
    for col in eff_dict:
      df_eff[col] = eff_dict[col]
    if True: 
        df_eff.to_csv(os.path.join(data_dic.eff_dir_path, data.file_name + '.csv'), index = None)
        print('Effective value save is done.')

# ---------------------------------------------------------------------------------------------------------------
for fi_num, fi_ in enumerate(filenames):
    
    if fi_num == 0: 
        # dirs how many dir in root path
        for dir_num, dir_ in enumerate(dirs):
            # creste dir path (root path + dir name) for class data_pre data_to_df to know how many file(.tdms) in this dir path
            dir_path = data_dic.rpath
            file     = [f for f in os.listdir(path = dir_path) if f.endswith(data_dic.ext[0])]
            for loop_num, i in enumerate(range(64,65)):
                # try:
                    data_pre.file_id = i
                    data = data_pre(
                        dir_path   = dir_path,
                        csv_s_path = data_dic.csv_dir_path, 
                        pic_s_path = data_dic.pic_dir_path,
                        eff_s_path = data_dic.eff_dir_path,
                        data_ext   = ext,
                        verbose    = True,
                        save       = False
                        )
                    data.data_to_df()
                    
                    strom    = data.data.loc[:,'Schweissstrom']
                    spannung = data.data.loc[:,'Schweissspannung']
                    t        = data.data.Zeit
                    # ------------------------------------------------------------------------------------------------------------------
                    data_eff = effectiv_trans(data = strom,order= 10,threshold = 2.1 )
                    new_begin_id, new_end_id, force_0, force_e = data_eff.ID_search(forward=1, backward= 1, squeezetime = data.squeezetime, holdtime = data.holdtime)
                    U        = effectiv_trans(data = spannung, begin_id = new_begin_id, end_id = new_end_id)
                    # force_e = int(force_e + round(data.holdtime*100/2))
                    force_e = int(force_e)
                    # force_0 = int(force_0 + round(data.squeezetime*100/8))
                    force_0 = int(new_begin_id)

                    I_rms = data_eff.effectiv_wert(id_search = True,EGM_window_width = EGM_fensterbreite)
                    I_rms_sort = np.sort(I_rms.values)
                    delta_I = I_rms_sort[-1] - np.median(I_rms_sort[I_rms_sort!=0])

                    U_rms = U.effectiv_wert(id_search = False,valley_id = data_eff.valley_id ,EGM_window_width = EGM_fensterbreite)
                    R     = U_rms/I_rms
                    P     = I_rms**2 * R 
                    P.fillna(0, inplace = True)
                    
                    valley_id = data_eff.valley_id

                    force_filt = effectiv_trans.kalman_filter_simple(data = data.data, col_num = 'Kanal 6 [Elektrodenkraft]' , threshold = 1.5, EGM = True, EGM_window_width = 800, verbose = True, Q = 5e-8, R = 0.1**2)
                    way_u_signal = effectiv_trans.kalman_filter_simple(data = data.data, col_num = 'Kanal 10 [Wegmessung Unten]', threshold = 2, EGM = True, EGM_window_width = 800,verbose= True, Q = 5e-8, R = 0.1**2)
                    way_o_signal = effectiv_trans.kalman_filter_simple(data = data.data, col_num = 'Kanal 9 [Wegmessung Oben]', threshold = 1.5, EGM = True, EGM_window_width = 800,verbose= True, Q = 5e-8, R = 0.1**2) 
                    if dir_num == loop_num == 0:
                          first_way_o = np.mean(way_o_signal[new_begin_id:new_end_id])
                    #------------------------------------------------------------------------------------------------------------------
                    F  = F_modell(t = t, force = force_filt, valley_id = valley_id)
                    s1 = F.F_rising_slope(cut_off = [0,1/1])
                    s2 = F.F_falling_slope(cut_off = [0,1/1], e = 0.03)
                    ##------------------------------------------------------------------------------------------------------------------         
                    sum_way           = way_o_signal + way_u_signal
                    tief_prozessende  = sum_way[force_e]- sum_way[force_0]
                    tief_schweissende = sum_way[new_end_id]- sum_way[force_0]
                    max_tief          = sum_way[force_e] - min(sum_way[new_begin_id:new_end_id])
                    ##-----------------------------------------------------------------------------------------------------------------
                    ## plot
                    ##-----------------------------------------------------------------------------------------------------------------
                    # fig, ax = plt.subplots(figsize =(10,7))

                    # ax.plot(t, way_u_signal,label = 'unten')
                    # ax.plot([t[force_0],t[force_e]],[sum_way[force_0],sum_way[force_e]],'x')
                    # ax.plot([t[new_begin_id],t[new_end_id]],[sum_way[new_begin_id],sum_way[new_end_id]],'x')
                    # ax.plot(t, I_rms*2, label = 'I')
                    # ax.plot(t, force_filt*2, label = 'F')
                    # ax.plot(t[valley_id], force_filt[valley_id], label = 'F_valley_id')
                    # ax.plot(t, way_o_signal,label = 'oben')
                    # ax.plot(t, sum_way,label = 'sum')
                    # ax.set_title(data.pointnumber)
                    # ax.legend()
                    # save_fig(data_dic.pic_dir_path, str(data.pointnumber),reselution = 200)
                    #------------------------------------------------------------------------------------------------------------------              
                    R_class = R_modell(MDK = 2,R_data = R, valley_id = valley_id, valley_id_sel = True, skiprows = 4, skipfooter = 10)
                    delta_R = R_class.change_point(width=40, cut_off=[0.15, 0.45], custom_cost=rpt.costs.CostL1(), jump=5, pen = 2, results_show = False, save_path=data_dic.pic_dir_path, fig_name= data.file_name ,title = data.file_name )
                    R10     = R_class.first_10_R_avg()
                    R90     = R_class.first_90_R_avg()
                    ser     = R_class.specific_electrical_resistance(R=R90, l = data.thickness, epe = tief_prozessende)
                    # ##------------------------------------------------------------------------------------------------------------------         
                    Q                  = Q_modell(time = t, power = P, valley_id = valley_id, time_interval = 40,MDK = 2)
                    Pv, delta_P, P_max = Q.Power_decrease_rate(Fs = data.samplerate, cutoff = 50, plot = False)
                    Q_total            = Q.Q_total()
                    t_diff             = Q.time_diff(begin_time = t[new_begin_id], percent = 0.5, cross_time=True)
                    Q_significa        = Q.Q_d_correlation_select()
                    Q_geschwin         = Q.Q_pdot
                    ##------------------------------------------------------------------------------------------------------------------         
                    data_list_dic_all[filenames[1]].append(list(I_rms))
                    data_list_dic_all[filenames[2]].append(list(U_rms))
                    data_list_dic_all[filenames[3]].append(list(R))
                    data_list_dic_all[filenames[4]].append(list(P))
                    data_list_dic_all[filenames[5]].append(list(way_o_signal))
                    data_list_dic_all[filenames[6]].append(list(way_u_signal))
                    data_list_dic_all[filenames[7]].append(list(force_filt))
                    data_list_dic_all[filenames[8]].append(list(valley_id))
                    data_list_dic_all[filenames[9]].append(list(Q.acc_interval_dQ))
                    data_list_dic_all[filenames[10]].append(Q_total)
                    data_list_dic_all[filenames[12]].append(t_diff)
                    data_list_dic_all[filenames[13]].append(Q_geschwin)
                    data_list_dic_all[filenames[14]].append(Q_significa)
                    data_list_dic_all[filenames[17]].append(delta_R)
                    data_list_dic_all[filenames[18]].append(R10)
                    data_list_dic_all[filenames[20]].append(Pv)
                    data_list_dic_all[filenames[21]].append(delta_P)
                    data_list_dic_all[filenames[22]].append(P_max)
                    data_list_dic_all[filenames[23]].append(tief_prozessende)
                    data_list_dic_all[filenames[24]].append(tief_schweissende)
                    data_list_dic_all[filenames[25]].append(max_tief)
                    data_list_dic_all[filenames[26]].append(ser)
                    data_list_dic_all[filenames[28]].append(R90)
                    data_list_dic_all[filenames[29]].append(data.weldingcurrent)
                    data_list_dic_all[filenames[30]].append(s1)
                    data_list_dic_all[filenames[31]].append(s2)
                    data_list_dic_all[filenames[32]].append(F.delta_F)
                    data_list_dic_all[filenames[33]].append(delta_I)
                    ##------------------------------------------------------------------------------------------------------------------ 
                    row_dic =  dict([(key.split('_')[-1], value[loop_num]) for key,value in data_list_dic_all.items() if len(value) != 0][9:])
                    ##------------------------------------------------------------------------------------------------------------------ 
                    df_feature_all = df_feature_all.append(row_dic,ignore_index=True)

                    if True:
                        # --------------------------------------------------------------------------------------------------------
                        ##save eff. data in order
                        # eff_write(time = t, I_rms = I_rms, U_rms= U_rms, R = R, P = P, OY = way_o_signal, UY = way_u_signal, Force = force_filt)
                        # --------------------------------------------------------------------------------------------------------
                        if dir_num == loop_num == 0:

                            d_write = data_pre.write_csv(
                                    w_path        = data_dic.train_dir_path, 
                                    file_name     = '{}.dat'.format(fi_),
                                    title         = title_dic[fi_],
                                    welding_condi = [data.electrodeforce, data.weldingtime, data.weldingcurrent],
                                    sheet_info    = [data.sheet1, data.sheet2, data.thickness],
                                    )
                            d_write.write('Punktnummer' + '\t' + 'dp1' + '\t' + 'dp2' + '\t' + 'dw' + '\n' 
                                + ' ' + '\t' + '[mm]'+ '\t' + '[mm]'+ '\t' + '[mm]' + '\n')
                            
                        try:
                            dp1, dp2, dw = data.dp1, data.dp2, data.dw

                            d_write.write(r'{:4d}'.format(data.pointnumber) + '\t' + '{:.2f}'.format(dp1) + '\t' + '{:.2f}'.format(dp2) + '\t' + '{:.3f}'.format(dw) + '\n')
                            
                            point_num_train.append(data.pointnumber)

                            data_list_dic_train[filenames[1]].append(list(I_rms))
                            data_list_dic_train[filenames[2]].append(list(U_rms))
                            data_list_dic_train[filenames[3]].append(list(R))
                            data_list_dic_train[filenames[4]].append(list(P))
                            data_list_dic_train[filenames[5]].append(list(way_o_signal))
                            data_list_dic_train[filenames[6]].append(list(way_u_signal))
                            data_list_dic_train[filenames[7]].append(list(force_filt))
                            data_list_dic_train[filenames[8]].append(list(valley_id))
                            data_list_dic_train[filenames[9]].append(list(Q.acc_interval_dQ))
                            data_list_dic_train[filenames[10]].append(Q_total)
                            data_list_dic_train[filenames[12]].append(t_diff)
                            data_list_dic_train[filenames[13]].append(Q_geschwin)
                            data_list_dic_train[filenames[14]].append(Q_significa)
                            data_list_dic_train[filenames[17]].append(delta_R)
                            data_list_dic_train[filenames[18]].append(R10)
                            data_list_dic_train[filenames[20]].append(Pv)
                            data_list_dic_train[filenames[21]].append(delta_P)
                            data_list_dic_train[filenames[22]].append(P_max)
                            data_list_dic_train[filenames[23]].append(tief_prozessende)
                            data_list_dic_train[filenames[24]].append(tief_schweissende)
                            data_list_dic_train[filenames[25]].append(max_tief)
                            data_list_dic_train[filenames[26]].append(ser)
                            data_list_dic_train[filenames[28]].append(R90)
                            data_list_dic_train[filenames[29]].append(data.weldingcurrent)
                            data_list_dic_train[filenames[30]].append(s1)
                            data_list_dic_train[filenames[31]].append(s2)
                            data_list_dic_train[filenames[32]].append(F.delta_F)
                            data_list_dic_train[filenames[33]].append(delta_I)

                            df_feature_train = df_feature_train.append(row_dic, ignore_index=True)

                        except AttributeError as error:
                            
                            print(error)

                            data_list_dic_test[filenames[1]].append(list(I_rms))
                            data_list_dic_test[filenames[2]].append(list(U_rms))
                            data_list_dic_test[filenames[3]].append(list(R))
                            data_list_dic_test[filenames[4]].append(list(P))
                            data_list_dic_test[filenames[5]].append(list(way_o_signal))
                            data_list_dic_test[filenames[6]].append(list(way_u_signal))
                            data_list_dic_test[filenames[7]].append(list(force_filt))
                            data_list_dic_test[filenames[8]].append(list(valley_id))
                            data_list_dic_test[filenames[9]].append(list(Q.acc_interval_dQ))
                            data_list_dic_test[filenames[10]].append(Q_total)
                            data_list_dic_test[filenames[12]].append(t_diff)
                            data_list_dic_test[filenames[13]].append(Q_geschwin)
                            data_list_dic_test[filenames[14]].append(Q_significa)
                            data_list_dic_test[filenames[17]].append(delta_R)
                            data_list_dic_test[filenames[18]].append(R10)
                            data_list_dic_test[filenames[20]].append(Pv)
                            data_list_dic_test[filenames[21]].append(delta_P)
                            data_list_dic_test[filenames[22]].append(P_max)
                            data_list_dic_test[filenames[23]].append(tief_prozessende)
                            data_list_dic_test[filenames[24]].append(tief_schweissende)
                            data_list_dic_test[filenames[25]].append(max_tief)
                            data_list_dic_test[filenames[26]].append(ser)
                            data_list_dic_test[filenames[28]].append(R90)
                            data_list_dic_test[filenames[29]].append(data.weldingcurrent)
                            data_list_dic_test[filenames[30]].append(s1)
                            data_list_dic_test[filenames[31]].append(s2)
                            data_list_dic_test[filenames[32]].append(F.delta_F)
                            data_list_dic_test[filenames[33]].append(delta_I)

                            df_feature_test = df_feature_test.append(row_dic, ignore_index=True)
                            
                            point_num_test.append(data.pointnumber)

                    data_eff.effectiv_plot(
                        data          = data.data,
                        I_rms         = I_rms,
                        U_rms         = U_rms,
                        R             = R,
                        P             = P,
                        data_name     = data.file_name,
                        save_pic      = True,
                        plot_pic      = True,
                        zoom          = False,
                        save_pic_path = data_dic.pic_dir_path 
                        )     

                    point_num_all.append(data.pointnumber)
                    print('es klappt')
                    print('\n')

                
                except:
                    print('error')

        df_feature_all.insert(0, 'pktnum', point_num_all)
        df_feature_test.insert(0, 'pktnum', point_num_test)
        df_feature_train.insert(0, 'pktnum', df_feature_train)
        df_feature_all.to_csv(data_dic.all_dir_path + 'test_data_feature_all.csv', index = None)
        df_feature_test.to_csv(data_dic.test_dir_path + 'feature_test.csv', index = None)
        df_feature_train.to_csv(data_dic.train_dir_path + 'feature_train.csv', index = None)

    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:

            train_data = executor.submit(
                data_write,
                **{
                    'data':data,
                    'w_path':data_dic.train_dir_path,
                    'title_dic':title_dic,
                    'filename':fi_, 
                    'filenames_list_dic':data_list_dic_train,
                    'point_num':point_num_train,
                    'add_time':True,
                    'title_unit_dic': title_unit,
                    'time':t
                   }
                )

            test_data = executor.submit(
                data_write,
                **{
                    'data':data,
                    'w_path':data_dic.test_dir_path,
                    'title_dic':title_dic,
                    'filename':fi_, 
                    'filenames_list_dic':data_list_dic_test,
                    'point_num':point_num_test,
                    'add_time':True,
                    'title_unit_dic': title_unit,
                    'time':t
                   }
                )

            all_data = executor.submit(
                data_write,
                **{
                    'data':data,
                    'w_path':data_dic.all_dir_path,
                    'title_dic':title_dic,
                    'filename':fi_, 
                    'filenames_list_dic':data_list_dic_all,
                    'point_num':point_num_all,
                    'add_time':True,
                    'title_unit_dic': title_unit,
                    'time':t
                   }
                )