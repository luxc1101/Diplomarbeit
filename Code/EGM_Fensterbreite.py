import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler 
from mpl_toolkits.axes_grid1 import make_axes_locatable

from labvtocsv2 import data_pre
from eff_trans2 import effectiv_trans
from plot import plot_effectiv
from figsave import save_fig
# ---------------------------------------------------------------------------------------------------------------
# Dieses Code zur Ermittlung, wie Breite des Fensters von EGM Method für Filtern des Stromsignal und
# des Sapnnungssignal sowie weiter berechnete Prozesswiderstand und  Prozessleistung kann die beste Effect erreichen
# ---------------------------------------------------------------------------------------------------------------
sv       = 'Fensterbreite'
r_path   = 'E:/Prozessdaten/MDK1/TDMS_datei/{}/'.format(sv)
ext      = '.tdms'
dirs     = [d for d in os.listdir(path = r_path) if os.path.isdir(os.path.join(r_path,d))]
dir_path = os.path.join(r_path,dirs[0])
# EGM_fensterbreite_user = 50
EGM_fensterbreite = 1
print(dirs)
Err_df = pd.DataFrame()
pkt_num = []
scaler = MinMaxScaler()

fig, ax = plt.subplots(figsize=(12,5))

font = {'family': 'serif',
        'size': 14
        }
prop={'family': 'serif', 'size':12}
# ------------------------------------------------------
for k in range(1):
    dir_path = os.path.join(r_path,dirs[k])
    # csv_save_path = os.path.join(dir_path,dirs[k]+'_CSV')
    file = [f for f in os.listdir(path = dir_path) if f.endswith(ext)]
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
        print(data.pointnumber)
        pkt_num.append(data.pointnumber)
        strom = data.data.iloc[:,8]
        t     = data.data.Zeit
        I     = effectiv_trans(data = strom,order= 10,threshold = 2.1)
        I_rms = I.effectiv_wert(id_search = True,EGM_window_width = EGM_fensterbreite)
        # plt.plot(strom)
        # plt.plot(I_rms)

        # plt.show()
        

        new_begin_id, new_end_id,a,b = I.ID_search(forward=1, backward= 1)
        peak_id = I.find_peak_id()
        valley_id = I.find_valley_id()

        min_ab_id = I_rms[valley_id[5:15]].idxmin()
        plt.plot(t[new_begin_id:new_end_id],I_rms[new_begin_id:new_end_id])
        plt.scatter(t[min_ab_id],I_rms[min_ab_id], marker='x',color = 'r', s = 100)

        new_begin_id = valley_id[100]
        new_end_id = valley_id[-2]
        # plt.scatter(t[new_begin_id],I_rms[new_begin_id], marker='x',color = 'k', s = 100)
        # plt.show()
        Bias = []
        Err  = []
        Err2 = []
        Var  = []
        AB   = []
        min_ab_l = []
        mse_l = []

        I_rms_user_df = pd.DataFrame(I_rms)

        EF_arr = np.arange(2,150,1)
        for EGM_fensterbreite_user in EF_arr:
            I_rms_user = I.effectiv_wert(id_search = True, EGM_window_width = EGM_fensterbreite_user)
            
            # y_noise   = np.var(I_rms[new_begin_id:new_end_id])
            y_var     = np.var(I_rms_user[new_begin_id:new_end_id])
            # mse       = mean_squared_error(strom[new_begin_id:new_end_id],I_rms_user[new_begin_id:new_end_id])
            y_bias_sq = np.mean((I_rms[new_begin_id:new_end_id] - np.mean(I_rms_user[new_begin_id:new_end_id]))**2)
            min_ab    = 1-I_rms[min_ab_id]/I_rms_user[min_ab_id]
            y_Err     = y_bias_sq + y_var + min_ab**2

            I_rms_user_df = pd.concat([I_rms_user_df,I_rms_user],axis=1)
            
        #     y_Err2    = min_ab**2 + y_var
            # print(mse)
            Var.append(y_var)
            Err.append(y_Err)
            # mse_l.append(mse)
            Bias.append(y_bias_sq)
            min_ab_l.append(min_ab)
            AB.append(min_ab)

        nor_Err = scaler.fit_transform(np.array(Err).reshape(-1,1)).reshape(-1)
        Err_series = pd.Series(nor_Err)
        Err_df = pd.concat([Err_df,Err_series],axis=1)

#         # print(I_rms_user_df.shape)
Err_df.to_csv(r_path + 'Error' + '.csv', index = None)

Err_df = pd.read_csv(r_path + 'Error.csv').dropna(how = 'all')
# print(Err_df)
# EF_arr = np.arange(2,21,1)

interp = 'bilinear'
# # interp = 'nearest'
fig, ax = plt.subplots(figsize=(12,5))

font = {'family': 'serif',
#         'size': 14
        }
prop={'family': 'serif', 'size':12}


# fig.canvas.manager.window.move(0,0)
# im = ax.imshow(Err_df.values, origin='upper', interpolation=interp,aspect='auto')
# # ax.set_yticks(EF_arr)
# ax.set_xlabel('Schweißpunktnummer', fontdict = font)
# ax.set_ylabel('Fensterbreite', fontdict = font)
# ax.set_xticks(np.arange(200)[::20]) 
# ax.set_xlim(0,190) 
# # ax.set_xticklabels(pkt_num[::9], color="k",fontsize = 10) 
# ax.set_yticks(np.arange(len(EF_arr))[::2]) 
# ax.set_yticklabels(EF_arr[::2], color="k",fontdict = font) 
# ax.tick_params(axis='both', which='major', labelsize=14,direction='in')
# # ax.grid()
# # divider = make_axes_locatable(ax)
# # cax = divider.append_axes("right", size="1%", pad=0.5)
# cbar = fig.colorbar(im,ax = ax, pad = 0.03)
# # cbar = fig.colorbar(im,ax = ax)
# cbar.set_label('Kosten',fontdict = font)
# plt.grid(ls = ':', alpha = 0.8)
# save_fig(image_path = r_path,fig_name = 'Beste_Fensterbreite_200_2')
# plt.show()

        with plt.rc_context({'axes.autolimit_mode': 'round_numbers'}):
    
            fig = plt.figure(figsize=(14,8))

            ax1 = fig.add_subplot(223)
            ax1.plot(EF_arr, Err,color = 'gray',linestyle = '--')
            ax1.scatter(EF_arr, Err,c=Err, cmap = cm.viridis ) 
            idmin = Err.index(min(Err))
            print(idmin)
            ax1.scatter(EF_arr[idmin], Err[idmin], marker='x',color = 'red',s = 100, label = 'optimierte Fensterbreite' ) 
            ax1.set_xlim(1,max(EF_arr))
            ax1.yaxis.get_major_formatter().set_powerlimits((0,1))
            # ax1.text(EF_arr[idmin],(max(Err)+min(Err))/1.8,'optimierte Festerbreite: {}'.format(EF_arr[idmin]),horizontalalignment='left',fontdict = font)
            # ax1.text(EF_arr[idmin],max(Err),'y = (Bais$^2$ + Variance) + Abweichung$^2$',horizontalalignment='left',fontdict = font)
            # ax1.set_ylim(min(Err)*0.99,max(Err)*1.01
            ax1.tick_params(axis='both', which='major', labelsize=12,direction='in')
            ax1.set_xlabel('Festerbreite',fontdict = font)
            ax1.set_ylabel('Kosten', fontdict = font)
            ax1.set_title('Kosten',fontdict = font)
            ax1.legend(prop = prop)

            ax2 = fig.add_subplot(224)
            color_idx = np.linspace(0, 1, idmin+2)
            for i,color_id in enumerate(color_idx):
                if i == 0:
                    ax2.plot(t[valley_id[5]:valley_id[40]],I_rms[valley_id[5]:valley_id[40]],color=plt.cm.viridis(color_id),label = 'Fensterbreite 1')
                    ax2.scatter(t[min_ab_id],I_rms[min_ab_id], marker='^',color=plt.cm.viridis(color_id), s = 50, edgecolors = 'k',label = 'Vergleichspkt.')
                else:
                    ax2.plot(t[valley_id[5]:valley_id[40]],I_rms_user_df.iloc[valley_id[5]:valley_id[40],i], color=plt.cm.viridis(color_id),label = 'Fensterbreite {}'.format(i+1))
                    ax2.scatter(t[min_ab_id],I_rms_user_df.iloc[valley_id[5]:valley_id[40],i][min_ab_id], marker='^',color=plt.cm.viridis(color_id), s = 50, edgecolors = 'k')
            
            ax2.legend(prop = prop,ncol=1)
            ax2.tick_params(axis='both', which='major', labelsize=12,direction='in')
            ax2.set_xlabel('$t$ / $s$',fontdict = font)
            ax2.set_ylabel('$I_{eff}$ / $kA$',fontdict = font)
            # ax2.set_xlim(0.640, 0.66)
            ax2.set_title('Effektivwerte des Stromsignals',fontdict = font)

            ax3 = fig.add_subplot(231)
            ax3.plot(EF_arr, Bias,color = 'gray',linestyle = '--')
            ax3.scatter(EF_arr, Bias,c=Bias, cmap = cm.viridis)
            ax3.set_xlim(1,max(EF_arr))
            ax3.yaxis.get_major_formatter().set_powerlimits((0,1))
            # ax2.set_ylim(min(Bias),max(Bias))
            ax3.tick_params(axis='both', which='major', labelsize=12,direction='in')
            ax3.set_xlabel('Festerbreite',fontdict = font)
            ax3.set_title('Bias$^2$',fontdict = font)

            ax4 = fig.add_subplot(232)
            ax4.plot(EF_arr, Var,color = 'gray',linestyle = '--')
            ax4.scatter(EF_arr, Var, c=Var, cmap = cm.viridis)
            ax4.set_xlim(1,max(EF_arr))
            ax4.yaxis.get_major_formatter().set_powerlimits((0,1))
            # ax3.set_ylim(min(Var)*0.99,max(Var)*1.01)
            ax4.tick_params(axis='both', which='major', labelsize=12,direction='in')
            ax4.set_xlabel('Festerbreite',fontdict = font)
            ax4.set_title('Varianz',fontdict = font)

            ax5 = fig.add_subplot(233)
            ax5.plot(EF_arr, min_ab_l,color = 'gray',linestyle = '--')
            ax5.scatter(EF_arr, min_ab_l, c=min_ab_l, cmap = cm.viridis)
            ax5.set_xlim(1,max(EF_arr))
            ax5.yaxis.get_major_formatter().set_powerlimits((0,1))
            # ax4.set_ylim(min(min_ab_l)*0.99,max(min_ab_l)*1.01)
            ax5.tick_params(axis='both', which='major', labelsize=12,direction='in')
            ax5.set_xlabel('Festerbreite',fontdict = font )
            ax5.set_title('Abweichung',fontdict = font)

            fig.canvas.manager.window.move(0,0)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=None, hspace=0.3)
            save_fig(image_path = r_path,fig_name = 'Beste_Fensterbreite_einzel_2')
            plt.show()


