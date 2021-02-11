import numpy as np
import pandas as pd
import pathlib
import os
import matplotlib.pyplot as plt
import concurrent.futures
from scipy import integrate
from sklearn.metrics import mean_absolute_error
import matplotlib.cm as cm

from labvtocsv2 import data_pre, data_write
from eff_trans2 import effectiv_trans
from plot import plot_effectiv
from figsave import save_fig
# ---------------------------------------------------------------------------------------------------------------
# Dieses Code zur Ermittlung, bei welche Value Q man das beste filterte Signal durch Kalman Filter bekommen.
# Ob unter dir Hilfe EGM man bessere ergebnisse bekommen kann. 
# Q: process variance
# ---------------------------------------------------------------------------------------------------------------
sv                   = 'sv5_4000_PB'
r_path               = 'E:/Prozessdaten/MDK1/TDMS_datei/{}/'.format(sv)
s_path               = os.path.join(pathlib.PureWindowsPath(r_path).parents[1],'Output')
print(s_path)
ext                  = '.tdms'

dirs                 = [d for d in os.listdir(path = r_path) if os.path.isdir(os.path.join(r_path,d))]

filenames            = ('{}_d'.format(sv), '{}_I'.format(sv), '{}_U'.format(sv),'{}_R'.format(sv),'{}_P'.format(sv),'{}_OY'.format(sv),'{}_UY'.format(sv),'{}_F'.format(sv), '{}_Q'.format(sv))
title                = ('DURCHMESSER','SCHWEISSSTROM','SCHWEISSSPANNUNG','PROZESSWIDERSTAND','PROZESSLEISTUNG','WEGMESSUNG OBEN','WEGMESSUNG UNTEN','ELEKTRODENKRAFT','GES.WAERMEMENGE')
title_dic            = dict(zip(filenames,title))
filenames_list_all   = {}
filenames_list_train = {}
filenames_list_test  = {}

EGM_fensterbreite   = 6 # die beste Fensterbreite

# creat dir to save  
csv_dir_path   = os.path.join(s_path,sv + '_CSV')
pic_dir_path   = os.path.join(s_path,sv + '_PIC')
eff_dir_path   = os.path.join(s_path,sv + '_EFF')
train_dir_path = os.path.join(s_path,sv + '_TRAIN', '')
test_dir_path  = os.path.join(s_path,sv + '_TEST' , '')
all_dir_path   = os.path.join(s_path,sv + '_ALL'  , '')
os.makedirs(s_path,exist_ok         =True)
os.makedirs(csv_dir_path,exist_ok   =True)
os.makedirs(pic_dir_path,exist_ok   =True)
os.makedirs(eff_dir_path,exist_ok   =True)
os.makedirs(train_dir_path,exist_ok =True)
os.makedirs(test_dir_path,exist_ok  =True)
os.makedirs(all_dir_path,exist_ok   =True)

print(dirs)
print(s_path)
# legend_properties = {'weight':'bold', 'size': 5}
# ---------------------------------------------------------------------------------------------------------------
def signaltonoise(a, axis, ddof): 
    a = np.asanyarray(a) 
    m = a.mean(axis) 
    sd = a.std(axis = axis, ddof = ddof) 
    print(sd)
    return np.where(sd == 0, 0, m / sd)
# ---------------------------------------------------------------------------------------------------------------

for fi_num, fi_ in enumerate(filenames):
    
    if fi_num == 0: 
   
        for dir_num, dir_ in enumerate(dirs[1:2]):

            print(dir_)
            dir_path      = os.path.join(r_path,dir_)
            csv_save_path = os.path.join(csv_dir_path,dir_+'_CSV')
            pic_save_path = os.path.join(pic_dir_path,dir_+'_PIC')
            eff_save_path = os.path.join(eff_dir_path,dir_+'_EFF')
            file          = [f for f in os.listdir(path = dir_path) if f.endswith(ext)]
           
            for i in range(1):
                data_pre.file_id = i
                data = data_pre(
                    dir_path   = dir_path,
                    csv_s_path = None, 
                    pic_s_path = None,
                    eff_s_path = None,
                    data_ext   = [ext],
                    verbose    = True,
                    save       = False
                    )
                data.data_to_df()
                
                strom    = data.data.iloc[:,8]
                spannung = data.data.iloc[:,4]
                t        = data.data.Zeit
                w_U      = data.data.iloc[:,11]

                # way_o_signal = effectiv_trans.kalman_filter_simple(data = data.data, col_num = 10, threshold = 2, EGM = True, EGM_window_width = 800,verbose= True, Q = 5e-8, R = 0.1**2)
                # force_filt   = effectiv_trans.kalman_filter_simple(data = data.data, col_num = 7, threshold = 1.5, EGM = True, EGM_window_width = 800, verbose = True, Q = 5e-8, R = 0.1**2)
                var_l = []
                snr_l = []
                mse_l = []
                ges_l = []

                Q_l = np.logspace(-1,-20,20)*5
                Q_label = ['%.1g' % ele for ele in Q_l]
                print(Q_l)


                for k in Q_l:
                    print(k)

                    way_u_signal = effectiv_trans.kalman_filter_simple(data = data.data, 
                        col_num = 'Kanal 10 [Wegmessung Unten]', 
                        threshold = 2, 
                        EGM = False, 
                        verbose= True, 
                        Q = k, 
                        R = 0.1**2)


                    # way_u_signal = effectiv_trans.kalman_filter_simple(
                    #     data = data.data,
                    #     col_num = 11,
                    #     threshold = 2,
                    #     EGM = False,
                    #     verbose= True,
                    #     Q = k,
                    #     R = 0.1**2
                        # )
                    filenames_list_all.setdefault(k, []).append(way_u_signal)

                    var = np.var(way_u_signal)
                    snr = signaltonoise(way_u_signal, axis = 0, ddof = 0)
                    mse = mean_absolute_error(y_true = w_U, y_pred = way_u_signal)
                    ges = var  + mse
                    
                    var_l.append(var)
                    snr_l.append(snr)
                    mse_l.append(mse)
                    ges_l.append(ges)

                    print(mse)
                    print(snr)
                    print(var)

print(mse_l)
###################################
#    0.03041 Kalman ohne EGM      #
#    0.01926 Kalman mit  EGM      #
###################################
font = {'family': 'serif',
        'size': 14
        }
prop={'family': 'serif', 'size':12}

fig = plt.figure(figsize=(8,6))

left, bottom, width, height = 0.1, 0.28, 0.8, 0.4

n = 4
colors = cm.viridis(np.linspace(0,1,n))


ax1 = fig.add_axes([left, bottom, width, height])

# ax1.plot(np.arange(len(Q_l)),var_l, color = colors[0],lw =1)
# ax1.scatter(np.arange(len(Q_l)),var_l,label = 'var', color = colors[0], marker='s', edgecolors = 'k',s = 50)
ax1.plot(np.arange(len(Q_l)),var_l,'s--',label = 'var', color = colors[0])
# ax1.plot(np.arange(len(Q_l)),snr_l, color = colors[1],lw =1)
# ax1.scatter(np.arange(len(Q_l)),snr_l,label = 'snr', color = colors[1], marker ='o', edgecolors = 'k',s = 50)
ax1.plot(np.arange(len(Q_l)),snr_l,'o--',label = 'snr', color = colors[1])
# ax1.plot(np.arange(len(Q_l)),mse_l, color = colors[2],lw =1)
# ax1.scatter(np.arange(len(Q_l)),mse_l,label = 'mse', color = colors[2], marker = '^' , edgecolors = 'k',s = 50)
ax1.plot(np.arange(len(Q_l)),mse_l,'^--',label = 'mse', color = colors[2])
# ax.scatter(np.arange(len(Q_l)),ges_l,label = 'ges')
ax1.set_yticks(np.arange(0,11))
ax1.set_xticks(np.arange(len(Q_l))[::2])
ax1.set_xticklabels(Q_label[::2]) 
ax1.tick_params(axis='both', which='major', labelsize=12,direction='in')
ax1.set_xlabel('Q', fontdict = font)
ax1.legend(prop = prop, frameon = False)

left, bottom, width, height = 0.1, 0.05, 0.2, 0.15
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(t,w_U,label = 'OG',color = 'gray')
ax2.plot(t,filenames_list_all[Q_l[1]][0],label = 'filted',color = 'k')
ax2.set_xticks([])
ax2.set_yticks([])
# ax2.legend(fontdict = font)
ax2.set_xlabel('Q = {}'.format(Q_l[1]),fontdict = font)
# ax2.set_title('Q = {}'.format(Q_l[0]),fontdict = font)
# ax2.tick_pirection='in')

left, bottom, width, height = 0.3, 0.05, 0.2, 0.15
ax3 = fig.add_axes([left, bottom, width, height])
ax3.plot(t,w_U,label = 'OG',color ='gray')
ax3.plot(t,filenames_list_all[Q_l[3]][0],label = 'filted',color = 'k')
ax3.set_xticks([])
ax3.set_yticks([])
# ax3.legend(fontdict = font)
ax3.set_xlabel('Q = {}'.format(Q_l[3]),fontdict = font)
# ax3.set_title('Q = {}'.format(Q_l[7]),fontdict = font)

# way_u_signal = effectiv_trans.kalman_filter_simple(
#     data = data.data,
#     col_num = 11,
#     threshold = 2,
#     EGM = True,
#     verbose= True,
#     Q = 5e-8,
#     R = 0.1**2
#     )

# way_u_signal = effectiv_trans.kalman_filter_simple(data = data.data, 
#     col_num = 'Kanal 10 [Wegmessung Unten]', 
#     threshold = 2, 
#     EGM = True, 
#     EGM_window_width = 800,
#     verbose= True, 
#     Q = 5e-8, 
#     R = 0.1**2)

left, bottom, width, height = 0.5, 0.05, 0.2, 0.15
ax4 = fig.add_axes([left, bottom, width, height])
ax4.plot(t,w_U,label = 'Signal roh',color = 'gray')
ax4.plot(t,filenames_list_all[Q_l[7]][0],label = 'Kalman',color = 'k')
# ax4.plot(t,way_u_signal,label = 'Kalman+EGM',color = 'r',lw = 1.5,ls = '-.')
ax4.set_xticks([])
ax4.legend(prop = prop,bbox_to_anchor=(-0.5, 4.2) ,loc = 'upper center', frameon = False)
ax4.set_yticks([])
# ax4.legend(fontdict = font)
ax4.set_xlabel('Q = {}'.format(Q_l[7]),fontdict = font)

left, bottom, width, height = 0.7, 0.05, 0.2, 0.15
ax5 = fig.add_axes([left, bottom, width, height])
ax5.plot(t,w_U,label = 'Signal roh',color = 'gray')
ax5.plot(t,filenames_list_all[Q_l[9]][0],label = 'Kalman',color = 'k')
ax5.set_xticks([])
ax5.set_yticks([])
# ax5.legend(fontsize = 'xx-small')
ax5.set_xlabel('Q = {}'.format(Q_l[9]),fontdict = font)

left, bottom, width, height = 0.1, 0.7, 0.2, 0.15
ax6 = fig.add_axes([left, bottom, width, height])
ax6.plot(t,w_U,label = 'OG',color = 'gray')
ax6.plot(t,filenames_list_all[Q_l[11]][0],label = 'filted',color = 'k')
ax6.set_xticks([])
ax6.set_yticks([])
ax6.set_title('Q = {}'.format(Q_l[11]),fontdict = font)

left, bottom, width, height = 0.3, 0.7, 0.2, 0.15
ax7 = fig.add_axes([left, bottom, width, height])
ax7.plot(t,w_U,label = 'OG',color = 'gray')
ax7.plot(t,filenames_list_all[Q_l[13]][0],label = 'filted',color = 'k')
ax7.set_xticks([])
ax7.set_yticks([])
ax7.set_title('Q = {}'.format(Q_l[13]),fontdict = font)

left, bottom, width, height = 0.5, 0.7, 0.2, 0.15
ax8 = fig.add_axes([left, bottom, width, height])
ax8.plot(t,w_U,label = 'OG',color = 'gray')
ax8.plot(t,filenames_list_all[Q_l[17]][0],label = 'filted',color = 'k')
ax8.set_xticks([])
ax8.set_yticks([])
ax8.set_title('Q = {}'.format(Q_l[17]),fontdict = font)

left, bottom, width, height = 0.7, 0.7, 0.2, 0.15
ax9 = fig.add_axes([left, bottom, width, height])
ax9.plot(t,w_U,label = 'OG',color = 'gray')
ax9.plot(t,filenames_list_all[Q_l[19]][0],label = 'Kalman',color = 'k')
ax9.set_xticks([])
ax9.set_yticks([])
# ax9.legend(prop = legend_properties,loc = 4 ) 
ax9.set_title('Q = {}'.format(Q_l[19]),fontdict = font)

save_fig(image_path = s_path,fig_name = 'Kalmanfilter_Q_3')
plt.show()



