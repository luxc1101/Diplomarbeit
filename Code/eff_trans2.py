import numpy as np
import pandas as pd
from scipy import signal
import itertools
from kalmanfilter import KalmanFilter, KalmanFilter_simple
import sys
from scipy import integrate
from scipy.interpolate import interp1d
from tqdm.auto import tqdm
from sklearn.linear_model import LinearRegression
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
import ruptures as rpt
import math
import matplotlib.pyplot as plt
from figsave import save_fig
from kneed import KneeLocator

def conta_area(r = None, h0 = None, w0 = None, wi = None, mean_force = None):
    '''
    ----------------
    DESCRIPTION
    ----------------
    for the caculation of the theoretical area of the welding joint

    https://de.wikipedia.org/wiki/Kugelsegment

    ℎ_𝑖 = ℎ_0 + ∆𝑠/2; 
    𝑎_𝑖 = sqrt(2𝑟ℎ_𝑖 − ℎ_𝑖^2); 
    𝐴_𝑖 = 𝜋∙𝑎_𝑖^2
    ------------------
    PARAMETER:
    ------------------
    r: the welding joint radius SR2 [mm]
    h0: h0 = r – ∆l [mm] ∆l = 7.416 mm, ∆l is uptp the type of the welding joint
    w0: the frist measured way of upper welding joint
    wi: everytimes the measured way of upper welding joint
    mean_force: the mean force
    ------------------
    return:
    ------------------ 
    return: theoretical contact area of welding joint, welding joint change in length , and pressure of the welding joint
    '''
    d_s  = wi-w0
    hi   = h0 + d_s/2
    ai   = np.sqrt(2*r*hi - hi**2)
    Ai   = math.pi*ai**2
    Pres = mean_force*1e3/Ai
    return Ai, -d_s/2, Pres

#############################################################################################################

#CALSS EFFECTIVE VALUE CONVERSION

#############################################################################################################
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
    ID_search:      function
    effectiv_wert:      function
    gradient_filter:    staticmethod
    butter_lowpass_filter:  staticmethod
    kalman_filter_simple:  staticmethod
    ------------------
    ATTRIBUTE:
    ------------------
    valley_id: ndarray valley id
    preak_id: ndarray peak id   
    '''
    def __init__(self, data, begin_id = None, end_id = None,order = None ,threshold = None, col_name = None):
        
        self.data      = data
 
        self.threshold = threshold
        
        self.order     = order
        
        self.col_name  = col_name

        if begin_id == end_id == None:

            try:
                id_value      = [(idx, i) for idx, i in enumerate(self.data.values) if 2.5<=i<=3]
                # print(id_value)
            
                self.begin_id = id_value[0][0]
            
                self.end_id   = id_value[-1][0]

            except:
                print('current signal hat problem')
        
        else:
        
            self.begin_id = begin_id
        
            self.end_id   = end_id

    def forward_s(self):

        while True:
        
            if self.data[self.begin_id] < self.threshold:
        
                break
        
            elif self.begin_id <0:
        
                print('No results found for new begin id.')
        
                break
        
            self.begin_id -= 1

        self.begin_id = self.begin_id + 1


    def backward_s(self):
        
        while True:
        
            if self.data[self.end_id] < self.threshold:
        
                break
        
            elif self.end_id >= len(self.data):
        
                print('No results found for new end id. ')
        
                break
        
            self.end_id += 1

        self.end_id = self.end_id-1


    def ID_search(self, forward =1,backward = 1, squeezetime = 0, holdtime = 0):
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

            force_0 = self.begin_id - squeezetime*100
        
            return self.begin_id, self.end_id, force_0

        if (forward == 0) and (backward == 1):
        
            self.backward_s()

            force_e = self.end_id + holdtime*100
        
            return self.begin_id, self.end_id, force_e
        
        if (forward == 1) and (backward == 1):
        
            self.forward_s()
        
            self.backward_s()

            force_0 = self.begin_id - squeezetime*100

            force_e = self.end_id + holdtime*100
        
            return self.begin_id, self.end_id, force_0, force_e

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

                print('flatten')
        
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
        
            begin_id, end_id, force_0, force_e = self.ID_search(1,1)
        
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

    def effectiv_plot(self, data, I_rms, U_rms, R, P, data_name:str, save_pic:bool, plot_pic:bool, zoom:bool, save_pic_path:str):
        '''
        ----------------
        DESCRIPTION
        ----------------
        A function to plot the effectiv data:
        I_rms: effectiv current
        U_rms: effectiv voltage
        R: resistance 
        P: power 
        ----------------
        PARAMETER
        ----------------
        data_name - name of the current data
        peak_id - peak index of the current
        valley_id - valley index of the current
        I_rms - effectiv current
        U_rms - effectiv voltage
        R - resistance 
        p - power
        plot_pic - bool
        zoom - bool show local zoom of current data
        save_pic - bool save the pic or not
        save_pic_path - if save pic is True, please gave the save path 
        '''
        begin_id, end_id  = self.begin_id, self.end_id
        valley_id,peak_id = self.valley_id, self.peak_id

        font = {'family': 'serif',
                # 'color':  'darkred',
                'weight': 'normal',
                'size': 12
                }
        prop={'family': 'serif', 'size':10}

        with plt.rc_context({'axes.autolimit_mode': 'round_numbers'}):
            fig = plt.figure(constrained_layout=False,figsize=(10,6))
            gs = fig.add_gridspec(nrows=2,ncols = 2, wspace =0.2)
            t = data.Zeit[begin_id:end_id]
            ####current
            ax1 = fig.add_subplot(gs[0,0])
            ax1.plot(t, data.Schweissstrom[begin_id:end_id], color = 'silver', label = 'Schweißstrom roh')
            ax1.plot(t, I_rms[begin_id:end_id], color = 'k', label = 'effektiver Schweißstrom')
            # ax1.scatter(data.loc[:,'Zeit'][peak_id], data.loc[:,'Schweissstrom'][peak_id], color = 'red', edgecolors = 'k', marker = 'o', alpha = 0.5, label = 'Peak')
            # ax1.scatter(data.loc[:,'Zeit'][valley_id], data.loc[:,'Schweissstrom'][valley_id], color = 'green', edgecolors = 'k', marker = 's', alpha = 0.5, label = 'Tal')
            ax1.legend(loc = 'best',prop = prop)
            # ax1.legend(loc='upper center', bbox_to_anchor=(0.5,1.2),frameon=False,ncol=2,prop = prop)
            ax1.set_ylabel('Schweissstrom $I_{s}$ / $kA$',fontdict = font)
            ax1.set_xlim(1.1,1.9)
            # ax1.tick_params(axis='both',labelsize = 8,direction = 'in')
            # ax1.set_title('Schweißstrom',fontdict = font)
            # ax1.set_xlabel('Zeit $t$ / $s$',fontdict = font)
            ax1.tick_params(axis='both', which='major', labelsize=12,direction='in')
            ax1.grid()
            plt.setp(ax1.get_xticklabels(), visible=False)
            if zoom:
                with plt.rc_context({'axes.autolimit_mode': 'round_numbers'}):
                    axins = ax1.inset_axes((0.2, 0.2, 0.45, 0.3))
                    axins.plot(t,data.Schweissstrom[begin_id:end_id],color = 'silver')
                    axins.plot(t, I_rms[begin_id:end_id], color = 'k')
                    axins.scatter(data.loc[:,'Zeit'][peak_id], data.loc[:,'Schweissstrom'][peak_id], color = 'red', edgecolors = 'k', marker = 'o', alpha = 0.5)
                    axins.scatter(data.loc[:,'Zeit'][valley_id], data.loc[:,'Schweissstrom'][valley_id], color = 'green', edgecolors = 'k', marker = 's', alpha = 0.5)

                    zone_left = begin_id + 8000
                    zone_right = zone_left + 500

                    x_ratio = 0 
                    y_ratio = 0.05 

                    x_ = data.loc[:,'Zeit']

                    xlim0 = x_[zone_left]-(x_[zone_right] - x_[zone_left])*x_ratio
                    xlim1 = x_[zone_right]+(x_[zone_right] - x_[zone_left])*x_ratio

                    y     = data.loc[zone_left:zone_right,'Schweissstrom']
                    ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio-0.5
                    ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio+0.5

                    axins.set_xlim(xlim0, xlim1)
                    axins.set_ylim(ylim0, ylim1)

                    axins.set_xticks([])
                    axins.set_yticks([])

                    tx0 = xlim0
                    tx1 = xlim1 + 0.01
                    ty0 = ylim0
                    ty1 = ylim1
                    sx  = [tx0,tx1,tx1,tx0,tx0]
                    sy  = [ty0,ty0,ty1,ty1,ty0]
                    ax1.plot(sx,sy,"darkred",alpha = 0.5)

                    xy  = (xlim0,ylim0)
                    xy2 = (xlim0,ylim1)
                    con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
                         axesA=axins,axesB=ax1,color = 'darkred',alpha = 0.5)
                    axins.add_artist(con)

                    xy  = (xlim1,ylim0)
                    xy2 = (xlim1,ylim1)
                    con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
                         axesA=axins,axesB=ax1,color = 'darkred',alpha = 0.5)
                    axins.add_artist(con)

            # voltage       
            ax2 = fig.add_subplot(gs[1,0],sharex = ax1 )
            ax2.plot(t, data.Schweissspannung[begin_id:end_id],color = 'silver',label = 'Schweißspannung roh')
            ax2.plot(t, U_rms[begin_id:end_id],color = 'k',label = 'effektive Schweißspannung')
            # ax2.scatter(data.loc[:,'Zeit'][valley_id], data.loc[:,'Schweissspannung'][valley_id], color = 'green', edgecolors = 'k', marker = 's', alpha = 0.5, label = 'Tal')
            ax2.legend(loc = 'best',prop = prop)
            # ax2.legend(loc='upper center', bbox_to_anchor=(0.5,1.2),frameon=False,ncol=2,prop = prop)
            ax2.set_ylabel('Schweißspannung $U_{s}$ / $V$',fontdict = font)
            ax2.set_xlabel('Zeit $t$ / $s$',fontdict = font)
            # ax2.tick_params(axis='both',labelsize = 8,direction = 'in')
            ax2.tick_params(axis='both', which='major', labelsize=12,direction='in')
            # ax2.set_title('Schweißspannung',fontdict = font)
            ax2.grid()
            plt.setp(ax2.get_xticklabels(), visible=True)
            if zoom:
                with plt.rc_context({'axes.autolimit_mode': 'round_numbers'}):
                    axins2 = ax2.inset_axes((0.3, 0.2, 0.45, 0.3))
                    axins2.plot(t,data.Schweissspannung[begin_id:end_id],color = 'silver')
                    axins2.plot(t, U_rms[begin_id:end_id], color = 'k')
                    # axins2.scatter(data.loc[:,'Zeit'][peak_id], data.loc[:,'Schweissspannung'][peak_id], color = 'red', edgecolors = 'k', marker = 'o', alpha = 0.3)
                    axins2.scatter(data.loc[:,'Zeit'][valley_id], data.loc[:,'Schweissspannung'][valley_id], color = 'green', edgecolors = 'k', marker = 's', alpha = 0.5)

                    zone_left = begin_id + 8000
                    zone_right = zone_left + 500

                    x_ratio = 0 
                    y_ratio = 0.05 

                    x_ = data.loc[:,'Zeit']

                    xlim0 = x_[zone_left]-(x_[zone_right] - x_[zone_left])*x_ratio
                    xlim1 = x_[zone_right]+(x_[zone_right] - x_[zone_left])*x_ratio

                    y     = data.loc[zone_left:zone_right,'Schweissspannung']
                    ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio-0.5
                    ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio+0.5

                    axins2.set_xlim(xlim0, xlim1)
                    axins2.set_ylim(ylim0, ylim1)

                    axins2.set_xticks([])
                    axins2.set_yticks([])

                    tx0 = xlim0
                    tx1 = xlim1 + 0.01
                    ty0 = ylim0
                    ty1 = ylim1
                    sx  = [tx0,tx1,tx1,tx0,tx0]
                    sy  = [ty0,ty0,ty1,ty1,ty0]
                    ax2.plot(sx,sy,"darkred",alpha = 0.5)

                    xy  = (xlim0,ylim0)
                    xy2 = (xlim0,ylim0)
                    con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
                         axesA=axins2,axesB=ax2,color = 'darkred',alpha = 0.5)
                    axins2.add_artist(con)

                    xy  = (xlim1,ylim0)
                    xy2 = (xlim1,ylim0)
                    con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
                         axesA=axins2,axesB=ax2,color = 'darkred',alpha = 0.5)
                    axins2.add_artist(con)


            # resistance
            ax3 = fig.add_subplot(gs[0,1])
            ax3.plot(t,R[begin_id:end_id],color = 'k',label = 'dynamischer Widerstand')
            # ax3.legend(loc = 1,prop = prop,frameon=False)
            ax3.set_ylabel('dynamischer Widerstand $R$ / $mΩ$',fontdict = font)
            # ax3.text(max(t)*0.7,max(R[begin_id:end_id]),'$R = U_{eff} / I_{eff}$',fontdict=font)
            ax3.tick_params(axis='both',labelsize = 12,direction = 'in')
            ax3.set_xlim(1.1,1.9)
            # ax3.set_title('Prozesswiderstand',fontdict = font)
            # ax3.set_xlabel('Zeit $t$ / $s$',fontdict = font) 
            ax3.grid()
            plt.setp(ax3.get_xticklabels(), visible=False)
            # power
            ax4 = fig.add_subplot(gs[1,1],sharex = ax3)
            ax4.plot(t,P[begin_id:end_id],color = 'k',label = 'Prozessleistung')
            # ax4.legend(loc = 'best',prop = prop,frameon=False)
            ax4.set_ylabel('Prozessleistung $P$/ $kW$',fontdict = font)
            ax4.set_xlabel('Zeit $t$ / $s$',fontdict = font)
            # ax4.set_title('Prozessleistung',fontdict = font)
            # ax4.text(max(t)*0.7,max(P[begin_id:end_id]),'$P = I^{2}_{eff} · R$',fontdict=font)
            ax4.tick_params(axis='both',labelsize = 12,direction = 'in')
            ax4.grid()
            plt.setp(ax4.get_xticklabels(), visible=True)

            # fig.suptitle('{}'.format(data_name), fontdict = font,y = 0.95, fontsize = 10)

            if save_pic:
                save_fig(image_path = save_pic_path,fig_name = data_name)
                print(save_pic_path)

            if plot_pic:
                plt.show()



    @staticmethod
    def gradient_filter(data, col_num, threshold, verbose: bool):
        '''
        ----------------
        DESCRIPTION
        ----------------
        Für manche Signals können unplausible Werte existieren. Um glatte Signal zu bekomman,
        sollte man solche Werte beseitigen. Aber zuerst muss man wissen wo genau soche Werte sind.
        Wegen extremer Value bei solchen Werte befindet sich bestimmt extreme Gradient um die Punkte.
        Durch Gradient-analyse kann man herausfinden, wo genau solche Punkte sind, und zwar die IDs.

        Dann werden die extremen Value durch die Mittelwert von vorherige Wert und nächste Wert dieses Punktes ersetzt.
        ----------------
        PARAMETER
        ----------------
        data: data.Dateframe 
        col_mun: columns number
        threshold: the max grandiet allowed
        verbose: bool show info or not
        -----------------
        RETURN
        ----------------
        x_ : Filtered signal pandas Series
        '''

        x = data.loc[:,col_num]
        
        x_ = x.copy()
        
        x_dot = np.gradient(np.array(x_),edge_order=1)
        
        x_dot_abs = abs(x_dot)
        
        if max(x_dot_abs) > threshold:
        
            mask = x_dot_abs>threshold
        
            if verbose:
        
                print('Extreme gradient {} of {}'.format(x_dot_abs[mask], x.name))
        
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
    def butter_lowpass_filter(data,cutoff,Fs,order):
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
        
        y = signal.filtfilt(b,a,data)
        
        return y

    @staticmethod
    def kalman_filter_simple(data, col_num, threshold, R, Q, EGM: bool, EGM_window_width = 800, verbose = True):
        '''
        ----------------
        DESCRIPTION
        ----------------
        Simple implementation of a Kalman filter based on:
        http://www.cs.unc.edu/~welch/kalman/kalmanIntro.html
        ----------------
        PARAMETER
        ----------------
        data: data.Dateframe 
        col_mun: columns number
        threshold: the max grandiet allowed (parammeter for def gradient_filter)
        R: Observation noise covariance 0.1**2
        Q: Process noise covariance 5e-8
        EGM: bool
        EGM_window_width: EGM windows width (800)
        verbose: True
        -----------------
        RETURN
        ----------------
        y : ndarray Filtered signal
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

            return np.array(EGM)

        else:

            return X_
#############################################################################################################

#CALSS Q MODELL

#############################################################################################################
class Q_modell:
    '''
    ------------------
    PARAMETER:
    ------------------
    time: time sereis
    power: the power
    valley_id: valley_id ndarray
    time_interval: decide how many parts of the heat signal are divided into
    ------------------
    POPULAR MEMBERS:
    ------------------
    Power_decrease_rate:    function
    Q_total:    function
    time_diff:  function
    Q_d_correlation_select: function
    Q_speed:    function
    ------------------
    ATTRIBUTE:
    ------------------
    '''

    def __init__(self, time, power, valley_id, time_interval, MDK):
        
        self.time          = time
        
        self.power         = power
        
        self.valley_id     = valley_id
        
        self.time_interval = time_interval 
        
        self.d_time        = np.array([self.time[self.valley_id[i+1]] - self.time[self.valley_id[i]] for i in range(len(self.valley_id)-1)])
        
        self.d_Q           = self.d_time * self.power[self.valley_id].values[:-1]

        self.MDK           = MDK
    
    def Power_decrease_rate(self, Fs, cutoff, plot:bool):

        t_ = self.time[self.valley_id].values

        P_by_lowpass = effectiv_trans.butter_lowpass_filter(data = self.power, cutoff = cutoff, Fs = Fs, order = 1)

        P_by_lowpass = P_by_lowpass[self.valley_id]

        P_by_lowpass_series = pd.Series(P_by_lowpass, index = t_)

        # detection the inflection point

        kl = KneeLocator(t_[-round(len(t_)*0.1):] ,P_by_lowpass[-round(len(P_by_lowpass)*0.1):], curve = "concave",direction="decreasing")

        if plot:

            kl.plot_knee()

            plt.show()

        inflection_pt_id = kl.knee

        max_id = P_by_lowpass_series.idxmax()

        self.P_max = max(P_by_lowpass_series)

        self.P_inflection = kl.knee_y

        delta_P = self.P_max - self.P_inflection

        delta_t = abs(max_id - inflection_pt_id)

        Pv = delta_P/delta_t

        Ps = self.P_max/delta_P

        return  Pv, delta_P, self.P_max

    def Q_total(self):

        Q = integrate.trapz(self.power.values, self.time.values)

        return Q

    def time_diff(self, begin_time, percent, cross_time:bool):
        '''
        ----------------
        DESCRIPTION
        ----------------
        Calculate the difference between the theoretical and actual time to reach a heat
        ----------------
        PARAMETER
        ----------------
        begin_time: the first time
        valley_id: valley_id ndarray
        percent: % 0...1
        cross_time: get the cross point coordination (x = time y = Q)
        ----------------
        RETURN
        ----------------
        time differenz: ttper - tpper like t_t50 - t_p50
        '''
        # self.d_time = np.array([self.time[self.valley_id[i+1]] - self.time[self.valley_id[i]] for i in range(len(self.valley_id)-1)])
        
        # self.d_Q    = self.d_time * self.power[self.valley_id].values[:-1]
        
        Q_p    = np.asarray(list(itertools.accumulate(self.d_Q,lambda x,y : x+y)))
        
        t_p    = np.asarray(list(itertools.accumulate(self.d_time,lambda x,y : x+y)))
        
        self.Q_p    = np.insert(Q_p,0,0)
        
        self.t_p    = np.insert(t_p,0,0) + begin_time
        
        self.t_tper = min(self.t_p) + (max(self.t_p)-min(self.t_p))*percent
        
        Q_tper = max(self.Q_p)*percent
        
        f      = interp1d(self.Q_p,self.t_p)
        
        self.t_pper = f(Q_tper)

        if cross_time:

            X = np.array([min(self.t_p),max(self.t_p)]).reshape(2,1)
            
            y = np.array([min(self.Q_p),max(self.Q_p)])

            reg = LinearRegression().fit(X,y)

            self.Q_t = reg.coef_[0]*self.t_p + reg.intercept_

            try:
                self.idx = np.argwhere(np.isclose(self.Q_p[:-1], self.Q_t[:-1], atol=5e-4)).reshape(-1)[1]    
                
                self.cross_time = (self.t_p[self.idx] - min(self.t_p))*1e3

                self.cross_Q = self.Q_p[self.idx]

            except:
                pass

        self.Q_tdot = Q_tper/(self.t_tper - min(self.t_p))

        self.Q_pdot = Q_tper/(self.t_pper - min(self.t_p))

        return self.t_tper - self.t_pper

    def Q_d_correlation_select(self):
        '''
        ----------------
        DESCRIPTION
        ----------------
        The heat signal is evenly divided into 30 parts
        Q 10 has the highest correlation in MDK1
        Q 30 has the highest correlation in MDK2
        ----------------
        PARAMETER
        ----------------
        MDK: 1 or 2

        ----------------
        RETURN
        ----------------
        Q sgf
        '''
        d_time = self.d_time*1e3
        
        acc_dt = np.asarray(list(itertools.accumulate(d_time,lambda x,y : x+y)))
        
        acc_interval_dt = np.array([sum(d_time[x:x+self.time_interval]) for x in range(0, len(d_time),self.time_interval)])
        
        d_Q = self.d_Q*1e3 # kJ->J

        self.acc_interval_dQ = np.array([sum(d_Q[x:x+self.time_interval]) for x in range(0, len(d_Q), self.time_interval)])

        if self.MDK == 1:

            Q_significa = self.acc_interval_dQ[9]

        if self.MDK ==2:

            Q_significa = self.acc_interval_dQ[29]

        return Q_significa

    def Q_speed_class(self, Q_t_dot, Q_p_dot, Q_tdot_l):
        '''
        ----------------
        DESCRIPTION
        ----------------
        Analyze the speed of heat. 
        check whether the speed is too fast or too slow
        ----------------
        PARAMETER
        ----------------
        Q_t_dot: theoretical heat speed
        Q_p_dot: actual heat speed
        Q_tdot_l: list to save the Q_t_dot and for each welding point to calculate the expected Q speed and upper limit lower limit  
        ----------------
        RETURN
        ----------------
        if Q speed > obere Grenze return 1
        if Q speed < untere Grenze return -1
        else return 0

        '''
        Q_tdot_l.append(Q_t_dot)
        
        Q_tdot_l_sum = sum(Q_tdot_l)
        
        self.erw = Q_tdot_l_sum/len(Q_tdot_l)
        
        self.o_grenz = self.erw * 1.2
        
        self.u_grenz = self.erw * 0.9

        if Q_p_dot > self.o_grenz:

            return 1

        elif Q_p_dot < self.u_grenz:
            
            return -1

        else:

            return 0    
#############################################################################################################

#CALSS R MODELL

#############################################################################################################
class R_modell:
    ''' 
    ------------------
    PARAMETER:
    ------------------
    R_data: resistance data, pandas sereis
    valley_id: the valley index of current
    valley_id_sel: by valley index selection can reduce data
    skiprows: number of lines to skip (int) at the start of the data
    skipfooter: number of lines at bottom of file to skip
    ------------------
    POPULAR MEMBERS:
    ------------------
    first_10_R_avg:    function
    change_point:  function
    ------------------
    ATTRIBUTE:
    ------------------
    valley_id: ndarray valley id
    R_data: pandas series 

    '''
    def __init__(self, MDK,R_data, valley_id = None, valley_id_sel = True ,skiprows = 0, skipfooter = 0):

        self.valley_id = valley_id

        self.MDK = MDK

        if valley_id_sel:

            R_data = R_data[self.valley_id]

            if skipfooter != 0:

                self.R_data = R_data.loc[(R_data.index>=self.valley_id[skiprows]) & (R_data.index<=self.valley_id[-skipfooter])] # pandas.Series

            else:

                self.R_data = R_data.loc[R_data.index>=self.valley_id[skiprows]]

        else:

            R_data = R_data.dropna()

            if skipfooter != 0:

                self.R_data = R_data.loc[(R_data.index>=self.valley_id[skiprows]) & (R_data.index<=self.valley_id[-skipfooter])] # pandas.Sereies

            else:

                self.R_data = R_data.loc[R_data.index>=self.valley_id[skiprows]]

    def first_10_R_avg(self):
        '''
        ----------------
        DESCRIPTION
        ----------------
        Calculate the top 10% average of resistance

        ----------------
        RETURN
        ----------------
        R10: float the mean value of top 10% resistance

        '''
        
        R_data_arr = self.R_data.values
        
        R10        = np.mean(R_data_arr[:round(len(R_data_arr)*0.11)])
        
        return R10

    def first_90_R_avg(self):
        '''
        ----------------
        DESCRIPTION
        ----------------
        Calculate the last 10% average of resistance

        ----------------
        RETURN
        ----------------
        R90: float the mean value of last 10% resistance

        '''
        
        R_data_arr = self.R_data.values
        
        R90        = np.mean(R_data_arr[-round(len(R_data_arr)*0.1):])
        
        return R90

    def specific_electrical_resistance(self, R, l, epe = None ,A0 = 28.31, Ai = None):
        '''
        ----------------
        DESCRIPTION
        ----------------
        Calculate the specific electrical resistance or volume resistivity

        ----------------
        PARAMETER
        ----------------
        R: the electrical resistance
        Ai: the cross-sectional area
        A0: default area at the begining
        l: the thickness of sheet
        epe: Impression at the end of the process
        MDK: 1 or 2
        ----------------
        RETURN
        ----------------
        ser: specific electrical resistance

        '''
        if self.MDK == 1:

            ser = R*Ai/l

        if self.MDK == 2:
            # if epe < 0 means no impression at the end of the process therfore epe = 0
            # if epe > 0 means there are impression therfore use this epe and can calculate the total remaining thickness

            epe = 0 if epe <0 else epe

            r_l = l - epe 

            ser = R*A0/r_l

        return ser


    def change_point(self, width:int, cut_off:list, custom_cost, jump:int, pen:float, results_show:bool, title = None, save_path = None, fig_name = None):
        '''                
        ----------------
        DESCRIPTION
        ----------------
        The purpose of the change point detection is to check whether there is a large enough sudden change 
        in a specific interval interval of the resistance signal.
        If there is a large enough change, it means that the explosion phenomenon has occurred during this welding.
        the algorithms of detection can be fund by this link:
        
        https://centre-borelli.github.io/ruptures-docs/index.html#documentation

        for the resistance data especially MDK2 this methode can be used to detective if a change point in selectarea
        if there is a change point, mean value before change point and after change point will be compared --> delta R
        else no change point delta R = 0
        because of material loss the dalta R musst bigger than 0, if the there is a chagne point but delta R < 0, 
        this situation has nothing to do with spritzer rarely occurs delta R can also be 0 
        and as usaual the resistance curve is going down with the time  
        ----------------
        PARAMETER
        ----------------
        width: int windows width 40
        cutoff: list [float, float], float: 0...1 1 means all data length will be selected [0.15, 0.45]
        custom_cost:  https://centre-borelli.github.io/ruptures-docs/costs/index.html
        jump: int subsample (one every jump points) 5
        pen:  float penalty value (>0) 2
        result_show : show image evaluation to displan the detective result
        title: the image title
        save_path: the path to save the result image
        fig_name: the image name
        ----------------
        RETURN
        ----------------
        delta_R: the Variation before and after the change point of resistance signal
        '''
        
        ab_R = self.R_data[round(len(self.valley_id)*cut_off[0]):round(len(self.valley_id)*cut_off[1])].values
        
        c    = custom_cost
        
        algo = rpt.Window(width = width, custom_cost = c, jump = jump).fit_predict(ab_R, pen = 2)
        
        if len(algo)>=2:
            delta_R = np.mean(ab_R[:algo[0]])-np.mean(ab_R[algo[0]:])
            if delta_R < 0:
                # delta_R can not less than 0 bescause the the resistance curve is going down with the time
                delta_R = 0
        else:
            delta_R = 0

        if results_show:
            rpt.display(ab_R, algo)
            if title != None:
                plt.title(title)
            if save_path and fig_name is not None:
                save_fig(image_path = save_path, fig_name = fig_name)
            plt.show()

        return delta_R
#############################################################################################################

#CALSS F MODELL

#############################################################################################################
class F_modell:
    '''   
    ------------------
    PARAMETER:
    ------------------
    t: time
    force: weldingforce
    valley_id: weldingcurrent valley index
    ------------------
    POPULAR MEMBERS:
    ------------------
    F_rising_slope: function get the rising slope
    F_falling_slope: function get the falling slope
    ------------------
    ATTRIBUTE:
    ------------------
    t: time
    force: weldingforce
    valley_id: valley index
    '''

    def __init__(self,t ,force, valley_id):
        
        self.t = t # pd.Sereis

        self.valley_id = valley_id #ndarray

        self.force = force #ndarray

    def F_rising_slope(self, cut_off:list ):
        '''
        ----------------
        DESCRIPTION
        ----------------
        in selected force data find the max force coordination
        calculate the rising slope between firt point and max force point of selected force data
        ----------------
        PARAMETER
        ----------------
        cutoff: list [float, float], float: 0...1 1 means all data length will be selected [0.15, 0.45]
        ----------------
        RETURN
        ----------------
        the rising slope
        '''
        self.force_cut = self.force[self.valley_id][round(len(self.valley_id)*cut_off[0]):round(len(self.valley_id)*cut_off[1])]
        
        self.t_cut     = self.t[self.valley_id].values[round(len(self.valley_id)*cut_off[0]):round(len(self.valley_id)*cut_off[1])]
        
        self.delta_F   = max(self.force_cut) - min(self.force_cut)
        
        self.max_id    = np.argmax(self.force_cut, axis=0)
        
        self.t0        = self.t_cut[0]

        self.tmax      = self.t_cut[self.max_id]
        
        self.F0        = self.force_cut[0]
        
        self.Fmax      = max(self.force_cut)
        
        s1 = (self.Fmax - self.F0)/(self.tmax - self.t0)
        
        return s1

    
    def F_falling_slope(self, cut_off:list, e = 0.03):
        '''
        ----------------
        DESCRIPTION
        ----------------
        in selected force data find the max force coordination
        calculate the rising slope between firt point and max force point of selected force data
        ----------------
        PARAMETER
        ----------------
        cutoff: list [float, float], float: 0...1 1 means all data length will be selected [0.15, 0.45].
        e: the time differenc dt between tmin and tmax. if dt > e calculate the absolute falling slope else the falling slope is 0
        ----------------
        RETURN
        ----------------
        the falling slope
        '''
        self.F_rising_slope(cut_off= cut_off)

        self.min_id = np.argmin(self.force_cut[self.max_id:], axis = 0)
        
        self.tmin   = self.t_cut[self.max_id:][self.min_id]
        
        self.Fmin   = min(self.force_cut[self.max_id:])
        
        dt = self.tmin - self.tmax

        if dt > e:

            s2 = abs((self.Fmin-self.Fmax)/(self.tmin - self.tmax))

        else:

            s2 = 0

        return s2








    






