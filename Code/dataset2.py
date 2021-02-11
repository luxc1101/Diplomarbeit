import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler,RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import os
import natsort
import re
import matplotlib.pyplot as plt
from matplotlib import cm
from figsave import save_fig
#############################################################################################
n = 25
colors = cm.viridis_r(np.linspace(0,1,n))

def data_set(DATA_PATH:str, zone:int,save:bool):
    feature_pb = pd.read_csv(os.path.join(DATA_PATH,pb_data)).set_index('pktnum').drop(columns = ['EL','ETS','dR'])
    feature_vb_pb = pd.read_csv(os.path.join(DATA_PATH,all_data)).set_index('pktnum').drop(columns = ['EL','ETS','dR'])
    feature_test = pd.read_csv(os.path.join(DATA_PATH,test_data)).set_index('pktnum').drop(columns = ['EL','ETS','dR'])
    pd_index = feature_pb.index
    vb_index = feature_vb_pb.index

    feature_pb = feature_vb_pb.loc[pd_index,:]
    feature_vb = feature_vb_pb.drop(index = pd_index)

    feature_vb_sel = feature_vb.loc[feature_vb.zone == zone]

    feature_vb_pb_sel =  pd.concat([feature_vb_sel, feature_pb], axis= 0).sort_index()

    if save:
        feature_pb.to_csv(os.path.join(save_path,'feature_pb.csv'))
        feature_vb_sel.to_csv(os.path.join(save_path,'feature_vb_sel.csv'))
        feature_vb_pb_sel.to_csv(os.path.join(save_path,'feature_vb_pb_sel.csv'))
        feature_test.to_csv(os.path.join(save_path,'feature_test.csv'))

    return feature_pb, feature_vb_sel,feature_vb_pb_sel,feature_vb_pb,feature_test


def feature_V_P(scaler,save):
    feature_pb, feature_vb_sel,feature_vb_pb_sel,feature_vb_pb,feature_test = data_set(data_path, 3, save)
    
    feature_vb_pb_sel.ETp.loc[feature_vb_pb_sel.ETp<0] = 0
    feature_vb_pb_sel.MET.loc[feature_vb_pb_sel.MET<0] = 0

    feature_vb_sel.ETp.loc[feature_vb_sel.ETp<0] = 0
    feature_vb_sel.MET.loc[feature_vb_sel.MET<0] = 0

    feature_pb.ETp.loc[feature_pb.ETp<0] = 0
    feature_pb.MET.loc[feature_pb.MET<0] = 0
    
    feature_test.ETp.loc[feature_test.ETp<0] = 0
    feature_test.MET.loc[feature_test.MET<0] = 0

    feature_vb_pb_sel = feature_vb_pb_sel.drop(columns = 'zone')
    feature_pb = feature_pb.drop(columns = 'zone')
    feature_vb_pb = feature_vb_pb.drop(columns = 'zone')
    feature_vb_sel = feature_vb_sel.drop(columns = 'zone')
    feature_test = feature_test.drop(columns = 'zone')

    scaler = scaler.fit(feature_vb_pb)

    feature_vb_pb_sel_nor = pd.DataFrame(scaler.transform(feature_vb_pb_sel), index = feature_vb_pb_sel.index, columns = feature_vb_pb_sel.columns)    
    feature_pb_nor = pd.DataFrame(scaler.transform(feature_pb), index = feature_pb.index, columns = feature_pb.columns)
    feature_vb_sel_nor = pd.DataFrame(scaler.transform(feature_vb_sel), index = feature_vb_sel.index, columns = feature_vb_sel.columns)
    feature_test_nor = pd.DataFrame(scaler.transform(feature_test), index = feature_test.index, columns = feature_test.columns)

    if save:
        feature_vb_pb_sel_nor.to_csv(os.path.join(save_path,'feature_vb_pb_sel_nor.csv'))
        feature_pb_nor.to_csv(os.path.join(save_path,'feature_pb_nor.csv'))
        feature_vb_sel_nor.to_csv(os.path.join(save_path,'feature_vb_sel_nor.csv'))
        feature_test_nor.to_csv(os.path.join(save_path,'feature_test_nor.csv'))

    return feature_pb_nor,feature_vb_sel_nor, feature_vb_pb_sel_nor,feature_test_nor


def data_set_dw(DATA_PATH,dsoll,save):
    feature_pb_nor, feature_vb_sel_nor,feature_vb_pb_sel_nor,feature_test_nor = feature_V_P(scaler,save)
    test_dw = pd.read_excel(os.path.join(data_path,'Torsionspruefung_MDK1.xlsx'))
    test_dw = test_dw.iloc[:-8].set_index('Punktnummer simuliert')
    test_dw = [1 if dm >= dsoll else 0 for dm in test_dw.dm_Korr.values]
    MDK1_all_dw  = pd.read_csv(os.path.join(DATA_PATH, dw), skiprows=12,sep='\t',header = None)
    MDK1_all_dw.columns = ['pktnum', 'dw1', 'dw2', 'dw']
    MDK1_all_dw = MDK1_all_dw[['pktnum', 'dw']].set_index('pktnum')
    feature_pb_nor['dw'] = MDK1_all_dw
    # feature_pb_nor_dw = feature_pb_nor.fillna(-1)
    
    feature_pb_nor.dw.loc[lambda x: x < dsoll] = 0
    feature_pb_nor.dw.loc[lambda x: x >= dsoll] = 1

    feature_pb_nor_x = feature_pb_nor[list(feature_pb_nor)[:-1]]
    feature_pb_nor_y = feature_pb_nor.dw.fillna(-1)


    return feature_pb_nor_x, feature_pb_nor_y, test_dw

def Labelspreading(DATA_PATH,dsoll,entropy,save,k):
    X_train, y_train = data_set_dw(DATA_PATH,dsoll,save)
    n_total_samples = len(y_train)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    n_labeled_points = n_total_samples-len(y_train.loc[y_train == -1])
    unlabeled_indices = y_train.loc[y_train == -1].index.values

    print("Label Spreading model: %d labeled & %d unlabeled (%d total)"% (n_labeled_points, n_total_samples - n_labeled_points, n_total_samples))
    i = 0
    max_entropy = []
    while True:
        if len(unlabeled_indices) == 0:
            print("No unlabeled items left to label.")
            break
        lp_model = LabelSpreading(gamma=10, max_iter=100, alpha=1e-10)
        lp_model.fit(X_train, y_train)
        predicted_labels = lp_model.transduction_[unlabeled_indices]
        print("Iteration %i %s" % (i, 70 * "_"))
        print("Label Spreading model: %d labeled & %d unlabeled (%d total)"% (n_labeled_points, 
                                                                              n_total_samples - n_labeled_points,
                                                                              n_total_samples))
        pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)
        # print(lp_model.label_distributions_.T[0]+lp_model.label_distributions_.T[1] )

        # print(lp_model.label_distributions_.T[0])
        # print(len(pred_entropies))

        # plt.plot(lp_model.label_distributions_.T[1],pred_entropies1,'o')

        print('max entropy {}'.format(max(pred_entropies[unlabeled_indices])))
        max_entropy.append(max(pred_entropies[unlabeled_indices]))
        if max(pred_entropies[unlabeled_indices]) <= entropy:
            print("Entropy littler than {}.".format(entropy))
            break
        uncertainty_index = np.argsort(pred_entropies)[::-1]
        uncertainty_index  = uncertainty_index[np.in1d(uncertainty_index, unlabeled_indices)][:1]
        print(uncertainty_index)
        delete_indices = np.array([], dtype=int)
        for index, class_index in enumerate(uncertainty_index):
            delete_index,  = np.where(unlabeled_indices == class_index)
            delete_indices = np.concatenate((delete_indices, delete_index))
        unlabeled_indices = np.delete(unlabeled_indices, delete_indices)
        X_train = X_train.drop(index = uncertainty_index ).reset_index(drop=True)
        y_train = y_train.drop(index = uncertainty_index ).reset_index(drop=True)
        n_total_samples -= len(uncertainty_index)
        # print(unlabeled_indices)

        unlabeled_indices = np.append(unlabeled_indices[unlabeled_indices<=uncertainty_index], unlabeled_indices[unlabeled_indices>=uncertainty_index]-1)
        i+=1

    y_train[unlabeled_indices] = predicted_labels
    print(predicted_labels)

    print('sum data with label %d'%len(y_train.values))
    MDK1_lbsp_nor = X_train
    MDK1_lbsp_nor['dw'] = y_train
    if save:
        MDK1_lbsp_nor.to_csv(os.path.join(save_path,'MDK1_lbsp_nor_{}.csv'.format(k)), index = None)

    return lp_model.label_distributions_.T[0][unlabeled_indices],lp_model.label_distributions_.T[1][unlabeled_indices], pred_entropies[unlabeled_indices]
    # ax.scatter(lp_model.label_distributions_.T[1][unlabeled_indices],pred_entropies[unlabeled_indices],marker='o', s = 8, label = '{},{}'.format(round(dsoll,2),len(unlabeled_indices)),alpha = 0.7)
    # g = ax.scatter(pkt_num,t_diff, marker='o', s = 8, cmap = cm.viridis,c = t_diff , label = 'zeitliche Differenz')
    # ax.set_facecolor('none')
##################################################################################################################
data_path = 'E:/Prozessdaten/MDK1/TDMS_datei/sv5_4000_all/Output_sv5_4000_VB/sv5_4000_VB_ALL/all_data_mdk1'
save_path = 'E:/Prozessdaten/MDK1/TDMS_datei/sv5_4000_all/Output_sv5_4000_VB/sv5_4000_VB_ALL/all_data_mdk1/data_set'
all_data  = 'all_vb_pb.csv'
pb_data   = 'feature_pb_MDK1.csv'
test_data = 'feature_Test_MDK1.csv'
dw        = 'sv5_4000_PB_d.dat'
test_dw   = 'Torsionspruefung_MDK1.xlse'

scaler    = MaxAbsScaler()
dsoll     = 4.9*np.sqrt(2)
dsoll_list = [4.8*np.sqrt(2),4.9*np.sqrt(2),5*np.sqrt(2),5.1*np.sqrt(2),5.2*np.sqrt(2)]
dsoll_list_str = ['$4,8 \sqrt{t}$','$4,9 \sqrt{t}$','$5,0 \sqrt{t}$','$5,1 \sqrt{t}$','$5,2 \sqrt{t}$']
font = {'family': 'serif',
        'size': 16
        }
prop={'family': 'serif', 'size':12}

##################################################################################################################
if __name__ == '__main__':
    feature_pb_nor, feature_vb_sel_nor,feature_vb_pb_sel_nor,feature_test_nor = feature_V_P(scaler,False)
    feature_pb_nor_x, feature_pb_nor_y,test_dw = data_set_dw(data_path,dsoll,False)
    fig, ax = plt.subplots(figsize=(6,5))
    marker_style = dict(linestyle=':', color='1', markersize=10,
                    mfc="C0", mec="C0")
    # marker_style.update(mec="None", markersize=15)
    # n = 5
    # colors = cm.viridis(np.linspace(0,1,n))
    # ax.axhline(0.7,0,1,ls = ':', color = 'r',label = '70% / 30%')
    # ax.axhline(0.3,0,1,ls = ':', color = 'r')
    # for i, dsoll in enumerate(zip(dsoll_list,dsoll_list_str)):
    #     x0,x1,y = Labelspreading(data_path,dsoll[0],0.6,True,dsoll[1][1:4])
    #     ax.scatter(np.arange(len(x0)),x0,marker="$0$",color = colors[i],s = 60,label = '{}, Anzahl: {}'.format(dsoll[1],len(y)))
    #     ax.scatter(np.arange(len(x1)),x1,marker="$1$",color = colors[i],s = 60)

    # ax.tick_params(axis='both', which='major', labelsize=14, direction='in')
    # ax.set_xlabel('Anzahl der neuen gelabelten Punkte', fontdict = font)
    # ax.set_ylabel('Wahrscheinlichkeit', fontdict = font)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5,1.19),frameon=False,ncol=3, prop = prop)
    # plt.grid()
    # # save_fig(image_path = 'C:/DA/Code/pywt/images/MDK1/ML', fig_name = 'labelspreading',reselution = 150)
    # plt.show()
    # print(feature_vb_pb_sel)