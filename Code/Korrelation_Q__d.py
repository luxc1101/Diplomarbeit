import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from figsave import save_fig

path = 'E:/Prozessdaten/MDK2/Output_schweissdata/Schweissdata_ALL/'
# path = 'E:/Prozessdaten/MDK1/TDMS_datei/Output_sv5_4000_PB/sv5_4000_PB_TRAIN/'
file_name = 'Schweissdata_PQ.dat'
# file_name = 'sv5_4000_PB_PQ.dat'

# path_d = 'E:/Prozessdaten/MDK1/TDMS_datei/Output_sv5_4000_PB/sv5_4000_PB_TRAIN/'
path_d = 'E:/Prozessdaten/MDK2/Output_schweissdata/Schweissdata_TRAIN/'
name_d = 'Schweissdata_d.dat'
# name_d = 'sv5_4000_PB_d.dat'
# s_path = 'C:/DA/Code/pywt/images'


font = {'family': 'serif',
        'size': 15
        }
prop={'family': 'serif', 'size':16}


data = pd.read_csv(path + file_name, skiprows=10, sep='\t').iloc[:,1:-1]
# print(data.head())

data_d = pd.read_csv(path_d + name_d, sep='\t', skiprows=12,header=None, index_col= 0)

dw_col = data_d.iloc[:,2].values

trian_id_int = list(data_d.index)
trian_id_str = [str(i) for i in trian_id_int]
print(trian_id_str)

data = data.loc[:,trian_id_str]
new_id = pd.Series(['$Q_{%d}$'%(i) for i in range(1,31)])
data.set_index(new_id, inplace = True)
print(data.head())
data_transposed = data.T
data_transposed['$d_w$'] = dw_col
# print(data_transposed)
# print(data_transposed.index)   
data_transposed.sort_index(inplace = True)
data_transposed.reset_index(drop = True,inplace = True)
# print(data_transposed)
# # print(data_transposed['$Q_{10}$'].corr(data_transposed['$d_w$']))
corr = data_transposed.corr()
y = corr['$d_w$'].iloc[:-1]
# print(y)

fig, ax = plt.subplots(figsize=(5,4))
width = 0.6
index = y.index
y_ = y.reset_index(drop = True)

ax.bar(y_.index, y_.values, width, color = 'gray')
ax.bar(y_.index[-1], y_.values[-1], width, color = 'darkred')
ax.set_ylim(0.7,0.8)
ax.set_xticks(np.arange(0,32,5))
ax.set_ylabel('Korrelationskoeffizient', fontdict = font)
ax.set_xlabel('partielle Wärmemenge Index', fontdict = font)
ax.set_xticklabels(index[::5], fontdict = font)
ax.tick_params(axis='y', which='major', labelsize=12,direction='in')
ax.tick_params(axis='x', rotation=45,which='major', labelsize=12)
ax.scatter(0,0,alpha = 0, label = '$Q_{i}$ - ges. Wärmemenge je 20 ms')
plt.legend(frameon = False, prop = prop)
save_fig(image_path = 'C:/DA/Code/pywt/images/MDK2/', fig_name = 'dQ_dw_2',reselution = 200)
plt.show()


# fig, ax = plt.subplots(figsize=(15,10))
# ax = sn.heatmap(data_transposed.corr(),ax = ax,annot = True,linewidths=.5,cbar_kws= {'label': 'Korrlationskoeffizient','aspect':40,'pad':0.01})
# ax.figure.axes[-1].yaxis.label.set_size(14)
# ax.tick_params(labelsize = 10)
# ax.set_ylabel('Parameternummer', fontsize = 14)
# ax.set_xlabel('Parameternummer', fontsize = 14)
# # save_fig(image_path=s_path, fig_name='Korrelation_Q_dw_2')
# plt.show()
# print(data_transposed.corr())
