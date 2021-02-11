import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from figsave import save_fig

from scipy import interpolate
from scipy import optimize
import math
from matplotlib.cbook import get_sample_data
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)

font = {'family': 'serif',
        'size': 14
        }
prop={'family': 'serif', 'size':13}
###############################################################################################################
data_path = 'E:/Prozessdaten/MDK1/'

ele_area  = pd.read_excel(data_path + 'Elektroden_AF.xlsx')

ele_area = ele_area.loc[ele_area.pktnum != 8]
print(ele_area.head())
print(ele_area.columns)

area_o    = ele_area['Arbeitsflaeche oben']
area_u    = ele_area['Arbeitsflaeche unten']

area_m    = ele_area['gemittelte Arbeitsflaeche']

delta_l_o = ele_area['delta Laenge oben']
delta_l_u = ele_area['delta Laenge unten']

delta_l_m = ele_area['gemittelte delta Laenge']

d_o       = ele_area['d oben']
d_u       = ele_area['d unten']

pktnum = ele_area.pktnum

maske      = ele_area.pktnum<2400
pktnum1    = pktnum[maske]
delta_l_u1 = delta_l_u[maske]
delta_l_o1 = delta_l_o[maske]

maske2 = ele_area.pktnum>=2200
pktnum2    = pktnum[maske2]
delta_l_u2 = delta_l_u[maske2]
delta_l_o2 = delta_l_o[maske2]

h0 = 0.584

print(pktnum1)
print(pktnum2)
xnew1 = np.linspace(pktnum1.values.min(),2200,1000)
xnew2 = np.linspace(2200,pktnum2.values.max(),1000)
xnew_all = np.linspace(0,4000,500)
parameter1 = np.polyfit(pktnum1.values, h0 + delta_l_u1.values, 3)
parameter2 = np.polyfit(pktnum2.values, h0 + delta_l_u2.values, 2)
parameter3 = np.polyfit(pktnum1.values, h0 + delta_l_o1.values, 3)
parameter4 = np.polyfit(pktnum2.values, h0 + delta_l_o2.values, 2)
p1 = np.poly1d(parameter1)
p2 = np.poly1d(parameter2)
p3 = np.poly1d(parameter3)
p4 = np.poly1d(parameter4)
print(p1)
print(p2)
print(p3)
print(p4)
def piecewise_linear_u(x, p1, p2):
    return np.piecewise(x, [x < 2300, x >= 2300],
        [lambda x: 3.332e-11*x**3 - 1.233e-07*x**2 + 0.000419*x + 0.5850,
         lambda x: -4.999e-08*x**2 + 0.0003831*x +  0.6856])

def piecewise_linear_o(x, p1, p2):
    return np.piecewise(x, [x < 2250, x >= 2250],
        [lambda x: 6.691e-11*x**3 - 2.136e-07*x**2 + 0.0004062*x + 0.613,
         lambda x: -3.076e-08*x**2 + 0.0002428*x + 0.8172])

pu , eu = optimize.curve_fit(piecewise_linear_u, pktnum.values, h0+delta_l_u.values)
po , eo = optimize.curve_fit(piecewise_linear_o, pktnum.values, h0+delta_l_o.values)

n = 4
colors = cm.viridis(np.linspace(0,1,n))

def auswerkung_k(data):
    return 2*math.pi*(8-data)

def delta_A(data):
    return auswerkung_k(data)*1e-3

oben = np.array([0.029 ,0.21025,0.24875,0.31775,0.345125,0.476375,0.543,0.55925,0.57525,0.54,0.57275]) + h0
unten = np.array([0.001, 0.096, 0.24375, 0.317375, 0.422, 0.53725, 0.595625, 0.609 ,0.6675,0.65275,0.630375]) + h0

Au = 100*delta_A(h0+delta_l_u.values)/area_u
Ao = 100*delta_A(h0+delta_l_o.values)/area_o
abs_u = abs(piecewise_linear_u(np.linspace(0,4000,11), *pu)-unten)
abs_o = abs(piecewise_linear_o(np.linspace(0,4000,11), *po)-oben)
einfluss_u = Au[::2]*(abs_u/1e-3)
einfluss_o = Ao[::2]*(abs_o/1e-3)

fig, host = plt.subplots(figsize = (8,4))
fig.subplots_adjust(right=0.8)
par1 = host.twinx()
width = 90
# l1 = host.scatter(pktnum, 100*delta_A(h0+delta_l_u.values)/area_u, marker = "o", edgecolor = 'k', s = 50,color = colors[0], label="Anteil unten")
# l2 = host.scatter(pktnum, 100*delta_A(h0+delta_l_o.values)/area_o, marker = "^", edgecolor = 'k', s = 50,color = colors[-1], label="Anteil oben")
# l2 = host.scatter(pktnum, h0+delta_l_u.values, marker = "v", edgecolor = 'k', s = 70,color = colors[0], label="$h_{i,unten}$",alpha = 0.9)
# l1 = host.scatter(pktnum, h0+delta_l_o.values, marker = "^", edgecolor = 'k', s = 70,color = colors[1], label="$h_{i,oben}$", alpha = 0.9)
# l1 = host.scatter(pktnum, auswerkung_k(h0+delta_l_o.values), marker = "v", edgecolor = 'k', s = 70,color = colors[0], label="$k_{i,unten}$", alpha = 0.9)
# l2 = host.scatter(pktnum, auswerkung_k(h0+delta_l_u.values), marker = "^", edgecolor = 'k', s = 70,color = colors[1], label="$k_{i,oben}$", alpha = 0.9)
# l1 = host.scatter(pktnum, delta_A(h0+delta_l_u.values), marker = "v", edgecolor = 'k', s = 70,color = colors[0], label="$\Delta A_{ab,i,unten}$", alpha = 0.9)
# l2 = host.scatter(pktnum, delta_A(h0+delta_l_o.values), marker = "^", edgecolor = 'k', s = 70,color = colors[1], label="$\Delta A_{ab,i,oben}$", alpha = 0.9)

# l1 = host.scatter(pktnum, delta_A(h0+delta_l_u.values), marker = "o", edgecolor = 'k', s = 50,color = colors[0], label="$\Delta A_{i}$ unten")
# l2 = host.scatter(pktnum, delta_A(h0+delta_l_o.values), marker = '^', edgecolor = 'k', s = 50,color = colors[-1], label="$\Delta A_{i}$ oben")
l3, = host.plot(xnew_all, piecewise_linear_u(xnew_all, *pu), label="gefittete $h_{i,unten}$",color = 'gray', ls = '--', lw = 2)
# l3, = host.plot(xnew_all, delta_A(piecewise_linear_u(xnew_all, *pu)), label="gefittete $\Delta A_{ab,i,unten}$ ",color = colors[0], ls = '--', )
# l3, = host.plot(xnew_all, auswerkung_k(piecewise_linear_u(xnew_all, *pu)), label="gefittete $k_{i,unten}$ ",color = colors[1], ls = '--', )
# l4, = host.plot(xnew_all, piecewise_linear_o(xnew_all, *po), label="gefittete $h_{i,oben}$ ", color =  colors[1], ls = ':', )
l4, = host.plot(xnew_all, piecewise_linear_o(xnew_all, *po), label="gefittete $h_{i,oben}$ ", color =  'gray', ls = ':', lw = 2)
# l4, = host.plot(xnew_all, delta_A(piecewise_linear_o(xnew_all, *po)), label="gefittete $\Delta A_{ab,i,oben}$ ", color =  colors[1], ls = ':', )
l5, = host.plot(np.linspace(0,4000,11), unten, 'v:',label="$h_{i,unten}$ ", color = colors[0],  lw = 2, markersize=8, mec = 'k')
l6, = host.plot(np.linspace(0,4000,11), oben, '^:',label="$h_{i,oben}$ ", color = colors[1],  lw = 2,markersize=8, mec = 'k')


bar_u = par1.bar(np.linspace(0,4000,11) - width/2, einfluss_u, width, label=r"$\alpha_{i,unten}$",color = colors[0],alpha = 0.7)
# bar_u = par1.bar(pktnum - width/2, 100*delta_A(h0+delta_l_u.values)/area_u, width, label=r"$\alpha_{i,unten}$",color = colors[0],alpha = 0.7)
bar_o = par1.bar(np.linspace(0,4000,11) + width/2, einfluss_o, width, label=r"$\alpha_{i,oben}$",color = colors[1],alpha = 0.7)
# bar_o = par1.bar(pktnum + width/2, 100*delta_A(h0+delta_l_o.values)/area_o, width, label=r"$\alpha_{i,oben}$",color = colors[1],alpha = 0.7)
# par1.set_yticks(np.arange(0,150,10))
# par1.set_ylim(0,100)
par1.tick_params(axis='both', which='major', labelsize=14, direction='in')
par1.set_ylabel(r"Anteil $\alpha_{i}$ / $\%$", fontdict = font)

props = dict(boxstyle='square', facecolor='wheat', alpha=0.4)
host.text(2500, 0.85, "$h_{i}$ = $h_{0}$ + $\Delta l$" , ha='center', fontsize=14,verticalalignment = 'bottom', bbox = props, fontdict = font)
# host.text(2000, 46, "$k_{i}$ = $2\pi(r-h_{i})$" , ha='center', fontsize=14,verticalalignment = 'bottom', bbox = props, fontdict = font)
par1.text(2500, 20, r"$\alpha_{i}$ = |$\Delta A_{i}$ / $A_{i}$|" , ha='center',verticalalignment = 'bottom', bbox = props,fontdict = font)
# print(max(delta_A(piecewise_linear_o(xnew_all, *po))),min(delta_A(piecewise_linear_o(xnew_all, *po))))
host.set_xticks(np.arange(0,4400,400))
host.set_xticklabels(np.arange(0,4400,400),rotation = '40')
host.set_xticklabels(np.arange(0,4400,400),rotation = '40')
par1.set_xlim(-80,4080)
host.set_yticks(np.arange(0.3,2,0.1))
# host.set_ylim(0.040,0.047)
host.set_ylim(0.5,1.5)
par1.set_ylim(0,100)
host.tick_params(axis='both', which='major', labelsize=14, direction='in')

host.set_xlabel("Schweißpunktnummer", fontdict = font)
host.set_ylabel("$h_{i}$ / $mm$", fontdict = font )
# host.set_ylabel("Auswirkungskoeffizient $k$ / $mm$", fontdict = font )
# host.set_ylabel(" $\Delta A_{ab,i}$ bei $\Delta h = 1e^{-3}$ / $mm^2$", fontdict = font )
host.grid()
host.ticklabel_format(axis = 'y',style = 'sci',scilimits = (3,2))

# with get_sample_data("C:/DA/Code/pywt/images/MDK1/Experiment_MDK1/hi.jpg") as file:
#     arr_img = plt.imread(file, format='jpg')

# imagebox = OffsetImage(arr_img, zoom=0.13)

# ab = AnnotationBbox(imagebox, (2940, 0.75),frameon = True)

# host.add_artist(ab)

lines = [l3,l4,l5,l6,bar_o,bar_u]
plt.gca().legend(lines, [l.get_label() for l in lines],loc='upper center', bbox_to_anchor=(0.5,1.25),frameon=False,ncol=3, prop = prop)
# save_fig(image_path = 'C:/DA/Code/pywt/images/MDK1/Experiment_MDK1', fig_name = 'elektroden_A%_hi_true',reselution = 200)
# save_fig(image_path = 'C:/DA/Code/pywt/images/MDK1/Experiment_MDK1', fig_name = 'elektroden_k', reselution = 300)
# save_fig(image_path = 'C:/DA/Code/pywt/images/MDK1/Experiment_MDK1', fig_name = 'elektroden_ai%', reselution = 300)
plt.show()

# print(100*delta_A(h0+delta_l_o.values)/area_o)

# x_test = np.linspace(0,4000,11)
# print(x_test)
# print(x_test)
# print(xnew_all[:10])
# print(piecewise_linear_o(x_test,*po)-0.584)
# print(piecewise_linear_u(x_test,*pu)-0.584)

###############################################################
# n = 4
# colors = cm.viridis(np.linspace(0,1,n))

# # colors2 = cm.jet(np.linspace(0,1,n))
# colors2 = cm.cividis(np.linspace(0,1,n))

# fig, host = plt.subplots(figsize = (8,4))
# fig.subplots_adjust(right=0.8)
# par1 = host.twinx()

# l1,   = host.plot(pktnum, area_o, "o:", color = colors[0], label="$A_{ab,oben}$", lw = 2)
# l2,   = host.plot(pktnum, area_u, "s:", color = colors[1], label="$A_{ab,unten}$", lw = 2)


# l3,   = par1.plot(pktnum, -1*delta_l_o, "^--", color = colors2[0], label="$\Delta L_{E,oben}$", )
# l4,   = par1.plot(pktnum, -1*delta_l_u, "v--", color = colors2[2], label="$\Delta L_{E,unten}$",)

# l5,   = par1.plot(pktnum, -1*delta_l_m, ls = ":", color = 'gray', label="gemittelte $\Delta L_{E}$")
# l6,   = host.plot(pktnum, area_m, ls = "-.", color = 'gray', label="gemittelte $A_{ab}$")


# host.set_xticks(np.arange(0,4400,400))
# host.set_xticklabels(np.arange(0,4400,400),rotation = '40')
# host.set_xlim(0,4000)
# host.set_yticks(np.arange(25,75,5))
# host.set_ylim(25,70)
# par1.set_yticks(np.arange(-0.9,1,0.1))
# par1.set_ylim(-0.9,0)
# host.tick_params(axis='both', which='major', labelsize=14, direction='in')
# par1.tick_params(axis='both', which='major', labelsize=14, direction='in')

# host.set_xlabel("Schweißpunktnummer", fontdict = font)
# host.set_ylabel("Elektrodenarbeitsfläche $A_{ab}$ / $mm^2$", fontdict = font)
# par1.set_ylabel("Elektrodenlängenänderung $-\Delta L_{E}$ / $mm$", fontdict = font)

# lines = [l1,l2,l3,l4,l5,l6]
# plt.gca().legend(lines, [l.get_label() for l in lines],loc='upper center', bbox_to_anchor=(0.5,1.2),frameon=False,ncol=3, prop = prop)
# save_fig(image_path = 'C:/DA/Code/pywt/images/MDK1/Experiment_MDK1', fig_name = 'elektroden_A_dl_true2',reselution = 300)
# plt.show()