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

from sklearn.metrics import mean_absolute_error, mean_squared_error

from labvtocsv2 import data_pre, data_write
from eff_trans2 import effectiv_trans, Q_modell, conta_area, R_modell, F_modell
from figsave import save_fig
from Fuzzy_logic import fuzzy_logic
import ruptures as rpt
from data_dict import data_dictionary
from brokenaxes import brokenaxes

# TensorFlow ≥2.0 is required
# import TENSORFLOW as tf
# from tensorflow import keras
# assert tf.__version__ >= "2.0"

# if not tf.config.list_physical_devices('GPU'):
#     print("No GPU was detected. LSTMs and CNNs can be very slow without a GPU.")
# ---------------------------------------------------------------------------------------------------------------
sv    = 'sv5_4000_plot'
# ext   = ['.dat','.dat_h']
ext   = ['.tdms']
# rpath = 'D:/Prozessdaten/MDK1/sv4_1600/'
# rpath = 'D:/Prozessdaten/MDK1/sv2_800'
# rpath = 'D:/Prozessdaten/MDK1/sv5_4000_PB/'
# rpath = 'D:/Prozessdaten/MDK1/sv1_400_PB/'
rpath = 'E:/Prozessdaten/MDK1/TDMS_datei/sv5_4000_all/sv5_4000/'
# rpath = 'E:/Prozessdaten/MDK1/Testdaten/insitu/'
# rpath = 'D:/Prozessdaten/MDK1/TDMS_datei/sv5_4000_PB/'
# rpath = 'L:/MDK1/Standmenge/sv5_4000/'

data_dic = data_dictionary(sv = sv, rpath = rpath, ext = ext)
data_dic.creat_dir(creat = True)

data_list_dic_all   = data_dic.data_save_dict()
data_list_dic_test  = data_dic.data_save_dict()
data_list_dic_train = data_dic.data_save_dict()

df_feature_all = data_dic.df_feature
df_feature_test = data_dic.df_feature
df_feature_train = data_dic.df_feature

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
dirsvb = [dir_ for dir_ in dirs if 'VB' in dir_] 
dirspb = [dir_ for dir_ in dirs if 'PB' in dir_]

font = {'family': 'serif',
                # 'color':  'darkred',
                'weight': 'normal',
                'size': 16
                }
prop={'family': 'serif', 'size':12}




# ---------------------------------------------------------------------------------------------------------------
for fi_num, fi_ in enumerate(filenames):
    if fi_num == 0:
        row_id = 0 
        # dirs how many dir in root path
        for dir_num, dir_ in enumerate(dirspb[:1]):
            
            print(dir_)
            # creste dir path (root path + dir name) for class data_pre data_to_df to know how many file(.tdms) in this dir path
            dir_path      = os.path.join(data_dic.rpath,dir_)
            csv_save_path = os.path.join(data_dic.csv_dir_path,dir_+'_CSV')
            pic_save_path = os.path.join(data_dic.pic_dir_path,dir_+'_PIC')
            eff_save_path = os.path.join(data_dic.eff_dir_path,dir_+'_EFF')
            file          = [f for f in os.listdir(path = dir_path) if f.endswith(data_dic.ext[0])]

            for loop_num, i in enumerate(range(1)):
              try:

                data_pre.file_id = i
                data = data_pre(
                    dir_path   = dir_path,
                    csv_s_path = csv_save_path, 
                    pic_s_path = pic_save_path,
                    eff_s_path = eff_save_path,
                    data_ext   = ext,
                    verbose    = True,
                    save       = False
                    )
                data.data_to_df()
                
                strom    = data.data.loc[:,'Schweissstrom']
                spannung = data.data.loc[:,'Schweissspannung']
                t        = data.data.Zeit
                #------------------------------------------------------------------------------------------------------------------
                data_eff = effectiv_trans(data = strom,order= 10,threshold = 2.2 ) #2.2/1
                new_begin_id, new_end_id, force_0, force_e = data_eff.ID_search(forward=1, backward= 1, squeezetime = data.squeezetime, holdtime = data.holdtime)
                U        = effectiv_trans(data = spannung, begin_id = new_begin_id, end_id = new_end_id)
                force_0_ = int(new_begin_id + round(data.squeezetime*100/3))
                # force_0_ = new_begin_id

                I_rms = data_eff.effectiv_wert(id_search = True,EGM_window_width = EGM_fensterbreite)
                U_rms = U.effectiv_wert(id_search = False,valley_id = data_eff.valley_id ,EGM_window_width = EGM_fensterbreite)
                R     = U_rms/I_rms
                P     = I_rms**2 * R 
                P.fillna(0, inplace = True)
                
                valley_id = data_eff.valley_id

                force_filt = effectiv_trans.kalman_filter_simple(data = data.data, col_num = 'Kanal 6 [Elektrodenkraft]' , threshold = 1.5, EGM = True, EGM_window_width = 800, verbose = True, Q = 5e-8, R = 0.1**2)
                way_u_signal = effectiv_trans.kalman_filter_simple(data = data.data, col_num = 'Kanal 10 [Wegmessung Unten]', threshold = 2, EGM = True, EGM_window_width = 800,verbose= True, Q = 5e-8, R = 0.1**2)
                way_o_signal = effectiv_trans.kalman_filter_simple(data = data.data, col_num = 'Kanal 9 [Wegmessung Oben]', threshold = 2, EGM = True, EGM_window_width = 800,verbose= True, Q = 5e-8, R = 0.1**2) 
                if max(way_o_signal) < 0:
                    way_o_signal_ = abs(way_o_signal)
                    way_o_signal  = 0.887*(abs(way_o_signal)/2)
                else:
                    way_o_signal_ = way_o_signal
                
                if dir_num == loop_num == 0:
                      first_way_o = np.mean(way_o_signal[new_begin_id:new_end_id])
                mean_way_o = np.mean(way_o_signal[new_begin_id:new_end_id])
                mean_force = np.mean(force_filt[new_begin_id:new_end_id])
                #------------------------------------------------------------------------------------------------------------------
                F  = F_modell(t = t, force = force_filt, valley_id = valley_id)
                s1 = F.F_rising_slope(cut_off = [0,1/1])
                #------------------------------------------------------------------------------------------------------------------
                sum_way           = way_o_signal_ + way_u_signal
                tief_prozessende  = sum_way[force_e]- sum_way[force_0_]
                tief_schweissende = sum_way[new_end_id]- sum_way[force_0_]
                max_tief          = sum_way[force_e] - min(sum_way[new_begin_id:new_end_id])
                ##---------------------------------------------------------------------------------------------------------
                ## plot 
                ##----------------------------------------------------------------------------------------------------------
                # fig = plt.figure(figsize=(5,4))
                # bax = brokenaxes(xlims=((0.664, 0.671), (1.265,1.274)),hspace=.04, despine=False)
                # id_value      = [(idx, i) for idx, i in enumerate(strom.values) if 5<=i<=6]
                # begin_id = id_value[0][0]
                # end_id   = id_value[-1][0]
                # # x = np.linspace(0, 1, 100)
                # bax.plot(t,strom.values, label = 'Stromsignal roh', color = 'gray', lw = 2)
                # bax.plot(t,I_rms,label = 'Effectivewert nach EGM', color = 'k', lw = 1.5)
                

                # bax.annotate('',
                #             xy=(t[new_begin_id]-0.001, strom.values[new_begin_id]), xycoords='data',
                #             xytext=(0.3,0.5), textcoords='axes fraction',
                #             arrowprops=dict(facecolor='black', shrink=0.05),
                #             horizontalalignment='right', verticalalignment='top')

                # bax.annotate('',
                #             xy=(t[new_end_id]+0.001, strom.values[new_end_id]), xycoords='data',
                #             xytext=(0.64,0.5), textcoords='axes fraction',
                #             arrowprops=dict(facecolor='black', shrink=0.05),
                #             horizontalalignment='right', verticalalignment='top')
                # sx  = [t[valley_id[4]],t[valley_id[-2]],t[valley_id[-2]],t[valley_id[4]],t[valley_id[4]]]
                # sy  = [strom.values[valley_id[-2]]-1,strom.values[valley_id[-2]]-1,strom.values[valley_id[-2]]+2,strom.values[valley_id[-2]]+2,strom.values[valley_id[-2]]-1]
                # bax.plot(sx,sy,"darkred",alpha = 0.5,lw = 2, label = 'Bereich zur EGM',ls = '--')

                # bax.axhline(strom.values[begin_id],0,1,color = 'k',ls = ':', label = 'threshold 1',lw = 1.5)
                # bax.axhline(strom.values[new_begin_id],0,1,color = 'k',ls = '-.',label = 'threshold 2',lw = 1.5)
                # bax.text(0.665, strom.values[end_id]+0.1, 'forward selection', rotation=80, fontdict = {'family': 'serif','weight': 'normal','size': 12})
                # bax.text(1.269, strom.values[end_id], 'backward selection', rotation=-70, fontdict = {'family': 'serif','weight': 'normal','size': 12})
                # bax.plot([t[begin_id],t[end_id]],[strom.values[begin_id],strom.values[end_id]], 's',color = 'blue',mec = 'k', alpha = 1, markersize = 8, label = 'Begin ID und Ende ID bei threshold 1')
                # bax.plot(t[valley_id],strom.values[valley_id], 'o',label = 'valley id', color = 'green',alpha = 0.5,mec = 'k',markersize = 8)
                # bax.plot(t[data_eff.peak_id],strom.values[data_eff.peak_id], 'o',label = 'peak id', color = 'r',alpha = 0.5,mec = 'k',markersize = 8)
                

                # bax.plot([t[new_begin_id],t[new_end_id]],[strom.values[new_begin_id],strom.values[new_end_id]], 'o',color = 'blue',mec = 'k', alpha = 1, markersize = 8, label ='Begin ID und Ende ID bei threshold 2')
                # # bax.legend(prop = prop)
                # bax.set_xlabel('Zeit in ms', fontdict =font)
                # bax.set_ylabel('Schweißstrom in kA', fontdict = font)
                # bax.tick_params(axis='both', which='major', labelsize=10, direction='in')
                # # bax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.18),frameon=False,ncol=2, prop = prop)
                # bax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.18),frameon=False,ncol=3, prop = prop)
                # save_fig(image_path = 'C:/DA/Code/pywt/images/MDK1/', fig_name = 'legend',reselution = 200 )


                # # bax.ticklabel_format(axis='x',style='sci',scilimits=(1,4))
                # plt.show()
                # plt.plot(t,strom)
                # plt.show()



                # fig, ax = plt.subplots(figsize =(10,7))

                # with plt.rc_context({'axes.autolimit_mode': 'round_numbers'}):
                #     fig = plt.figure(constrained_layout=False,figsize=(13,4))
                #     gs = fig.add_gridspec(nrows=1,ncols = 2, wspace =0.2)
                #     # kraft
                #     ax1 = fig.add_subplot(gs[0,0])
                #     ax1.plot(t, data.data.loc[:, 'Kanal 6 [Elektrodenkraft]'], color = 'silver', label = 'Elektrodenkraft roh')
                #     ax1.plot(t, force_filt, color = 'k', label = 'Elektrodenkraft gefiltert')
                #     ax1.set_xlim(0,max(t))
                #     ax1.legend(loc='upper center', bbox_to_anchor=(0.5,1.12),frameon=False,ncol=2,prop = prop)
                #     ax1.set_ylabel('Elektrodenkraft $F_{E}$ / $kN$',fontdict = font)
                #     ax1.set_xlabel('Zeit $t$ / $s$',fontdict = font)
                #     ax1.tick_params(axis='both', which='major', labelsize=12,direction='in')
                #     ax1.grid()

                #     ax2 = fig.add_subplot(gs[0,1],sharex = ax1)
                #     ax2.plot(t, data.data.loc[:, 'Kanal 9 [Wegmessung Oben]'],color = 'silver',label = 'Wegmessung roh')
                #     ax2.annotate('Wegmessung oben', xy=(0.55, 0.8), xycoords = 'axes fraction', xytext=(0.6, 0.9), ha="center", va="bottom", color = 'darkred',
                #         arrowprops=dict(arrowstyle="<-", color='darkred',lw = 1.5))
                #     ax2.annotate('Wegmessung unten', xy=(0.55, 0.3), xycoords = 'axes fraction', xytext=(0.6, 0.4), ha="center", va="bottom",color = 'darkred',
                #         arrowprops=dict(arrowstyle="<-", color='darkred', lw = 1.5))
                #     ax2.plot(t, data.data.loc[:, 'Kanal 10 [Wegmessung Unten]'],color = 'silver')
                #     ax2.plot(t, way_o_signal, color = 'k', label = 'Wegmessung  gefiltert')
                #     ax2.plot(t, way_u_signal, color = 'k',)
                #     ax2.legend(loc='upper center', bbox_to_anchor=(0.5,1.12),frameon=False,ncol=2,prop = prop)
                #     ax2.set_ylabel('Wegmessung $H_{E}$ / $mm$',fontdict = font)
                #     ax2.set_xlabel('Zeit $t$ / $s$',fontdict = font)
                #     ax2.tick_params(axis='both', which='major', labelsize=12,direction='in')
                #     ax2.grid()
                # ax.plot(t, data.data.loc[:,'Kanal 6 [Elektrodenkraft]'],label = 'kraft')
                # ax.plot([t[force_0],t[force_e]],[sum_way[force_0],sum_way[force_e]],'x',label = 'force_0, force_e')
                # ax.plot([t[force_0_],t[force_e]],[sum_way[force_0_],sum_way[force_e]],'x',label = 'force_0_, force_e')
                # ax.plot([t[new_begin_id],t[new_end_id]],[sum_way[new_begin_id],sum_way[new_end_id]],'x')
                # ax.plot(t, I_rms*2, label = 'I')
                # ax.plot(t, force_filt*2, label = 'F')
                # ax.plot(t[valley_id], force_filt[valley_id], label = 'F_valley_id')
                # ax.plot(t, way_o_signal,label = 'oben')
                # ax.plot(t, sum_way,label = 'sum')
                # ax.set_title(data.pointnumber)
                    
                    # save_fig(pic_save_path, 'weg_kraft')
                    # plt.show()
                ##------------------------------------------------------------------------------------------------------------------         
                Ai, elc_length, Pres = conta_area(r = 8, h0 = 0.584, w0 = first_way_o, wi = mean_way_o, mean_force = mean_force)
                # ##------------------------------------------------------------------------------------------------------------------         
                R_class = R_modell(R_data = R, valley_id = valley_id, valley_id_sel = True, skiprows = 4, skipfooter = 3, MDK = 1)
                delta_R = R_class.change_point(width=40, cut_off=[0.15, 0.45], custom_cost=rpt.costs.CostL1(), jump=5, pen = 2, results_show = False, save_path=pic_save_path, fig_name= data.file_name ,title = data.file_name )
                R10     = R_class.first_10_R_avg()
                R90     = R_class.first_90_R_avg()
                ser     = R_class.specific_electrical_resistance(R=R10, Ai = Ai, l = data.thickness)
                # ##------------------------------------------------------------------------------------------------------------------         
                Q                  = Q_modell(time = t, power = P, valley_id = valley_id, time_interval = 40, MDK = 1)
                Pv, delta_P, P_max = Q.Power_decrease_rate(Fs = data.samplerate, cutoff = 50, plot = False)
                Q_total            = Q.Q_total()
                t_diff             = Q.time_diff(begin_time = t[new_begin_id], percent = 0.5, cross_time=False)
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
                # data_list_dic_all[filenames[11]].append(mean_way_o)
                data_list_dic_all[filenames[12]].append(t_diff)
                data_list_dic_all[filenames[13]].append(Q.Q_pdot)
                data_list_dic_all[filenames[14]].append(Q_significa)
                data_list_dic_all[filenames[15]].append(Ai)
                data_list_dic_all[filenames[16]].append(Pres)
                data_list_dic_all[filenames[17]].append(delta_R)
                data_list_dic_all[filenames[18]].append(R10)
                data_list_dic_all[filenames[19]].append(elc_length)
                data_list_dic_all[filenames[20]].append(Pv)
                data_list_dic_all[filenames[21]].append(delta_P)
                data_list_dic_all[filenames[22]].append(P_max)
                data_list_dic_all[filenames[23]].append(tief_prozessende)
                data_list_dic_all[filenames[24]].append(tief_schweissende)
                data_list_dic_all[filenames[25]].append(max_tief)
                data_list_dic_all[filenames[26]].append(ser)
                data_list_dic_all[filenames[27]].append(data.zone)
                data_list_dic_all[filenames[28]].append(R90)
                data_list_dic_all[filenames[30]].append(s1)
                data_list_dic_all[filenames[32]].append(F.delta_F)
                ##------------------------------------------------------------------------------------------------------------------ 
                row_dic =  dict([(key.split('_')[-1], value[row_id]) for key,value in data_list_dic_all.items() if len(value) != 0][9:])
                ##------------------------------------------------------------------------------------------------------------------ 
                df_feature_all = df_feature_all.append(row_dic,ignore_index=True)

                if True:
                    # --------------------------------------------------------------------------------------------------------
                    ####save eff. data in order
                    # df_eff   = pd.DataFrame(t)
                    # eff_dict = dict(zip(data_list_dic_all.keys(),(I_rms,U_rms,R,P,way_o_signal,way_u_signal,force_filt)))
                    # for col in eff_dict:
                    #   df_eff[col] = eff_dict[col]
                    # if True: 
                    #     df_eff.to_csv(os.path.join(eff_save_path, data.file_name + '.csv'), index = None)
                    #     print('Effective value save is done.')
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
                        # data_list_dic_train[filenames[11]].append(mean_way_o)
                        data_list_dic_train[filenames[12]].append(t_diff)
                        data_list_dic_train[filenames[13]].append(Q.Q_pdot)
                        data_list_dic_train[filenames[14]].append(Q_significa)
                        data_list_dic_train[filenames[15]].append(Ai)
                        data_list_dic_train[filenames[16]].append(Pres)
                        data_list_dic_train[filenames[17]].append(delta_R)
                        data_list_dic_train[filenames[18]].append(R10)
                        data_list_dic_train[filenames[19]].append(elc_length)
                        data_list_dic_train[filenames[20]].append(Pv)
                        data_list_dic_train[filenames[21]].append(delta_P)
                        data_list_dic_train[filenames[22]].append(P_max)
                        data_list_dic_train[filenames[23]].append(tief_prozessende)
                        data_list_dic_train[filenames[24]].append(tief_schweissende)
                        data_list_dic_train[filenames[25]].append(max_tief)
                        data_list_dic_train[filenames[26]].append(ser)
                        data_list_dic_train[filenames[27]].append(data.zone)
                        data_list_dic_train[filenames[28]].append(R90)
                        data_list_dic_train[filenames[30]].append(s1)
                        data_list_dic_train[filenames[32]].append(F.delta_F)

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
                        # data_list_dic_test[filenames[11]].append(mean_way_o)
                        data_list_dic_test[filenames[12]].append(t_diff)
                        data_list_dic_test[filenames[13]].append(Q.Q_pdot)
                        data_list_dic_test[filenames[14]].append(Q_significa)
                        data_list_dic_test[filenames[15]].append(Ai)
                        data_list_dic_test[filenames[16]].append(Pres)
                        data_list_dic_test[filenames[17]].append(delta_R)
                        data_list_dic_test[filenames[18]].append(R10)
                        data_list_dic_test[filenames[19]].append(elc_length)
                        data_list_dic_test[filenames[20]].append(Pv)
                        data_list_dic_test[filenames[21]].append(delta_P)
                        data_list_dic_test[filenames[22]].append(P_max)
                        data_list_dic_test[filenames[23]].append(tief_prozessende)
                        data_list_dic_test[filenames[24]].append(tief_schweissende)
                        data_list_dic_test[filenames[25]].append(max_tief)
                        data_list_dic_test[filenames[26]].append(ser)
                        data_list_dic_test[filenames[27]].append(data.zone)
                        data_list_dic_test[filenames[28]].append(R90)
                        data_list_dic_test[filenames[30]].append(s1)
                        data_list_dic_test[filenames[32]].append(F.delta_F)

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
                        save_pic_path = pic_save_path
                        )     
                point_num_all.append(data.pointnumber + 8*dir_num)
                row_id += 1
                print('es klappt')
                print('\n')
              except:
                print('error')

        df_feature_all.insert(0, 'pktnum', point_num_all)
        df_feature_test.insert(0, 'pktnum', point_num_test)
        df_feature_train.insert(0, 'pktnum', point_num_train)
        df_feature_all.to_csv(data_dic.all_dir_path + 'feature_Test_MDK1.csv', index = None)
        df_feature_test.to_csv(data_dic.test_dir_path + 'feature_pb{}_test_MDK1.csv'.format(dir_.split('PB')[1]), index = None)
        df_feature_train.to_csv(data_dic.train_dir_path + 'feature_pb{}_train_MDK1.csv'.format(dir_.split('PB')[1]), index = None)

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

############################################################################################################################

