import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import seaborn as sns
from figsave import save_fig
import os
import seaborn as sns
# ------------------------------------------------------------------------------------
# PROJECT_ROOT_DIR = "."
# IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
# os.makedirs(IMAGES_PATH, exist_ok=True)
# ------------------------------------------------------------------------------------
data_path = 'C:/DA/Code/LVQ/Schweissdata.csv'

def fuzzy_logic(path, max_I, max_Q, max_Qdot, max_dI, strom, waerme, Q_dot):

    data = pd.read_csv('C:/DA/Code/LVQ/Schweissdata.csv', index_col=0)
    data.Zustand.replace({-1: 0, 0: 1, 1: 2 }, inplace = True)

    kalt = data.loc[data['Zustand'] == 0]
    normal = data.loc[data['Zustand'] == 1]
    spritzer = data.loc[data['Zustand'] == 2]

    kalt_I_sgm         = np.std(kalt[['Strom']])
    kalt_I_mu          = np.mean(kalt[['Strom']])
    normal_I_sgm       = np.std(normal[['Strom']])
    normal_I_mu        = np.mean(normal[['Strom']])
    spritzer_I_sgm     = np.std(spritzer[['Strom']])
    spritzer_I_mu      = np.mean(spritzer[['Strom']])

    kalt_Qpdot_sgm     = np.std(kalt[['Qpdot']])
    kalt_Qpdot_mu      = np.mean(kalt[['Qpdot']])
    normal_Qpdot_sgm   = np.std(normal[['Qpdot']])
    normal_Qpdot_mu    = np.mean(normal[['Qpdot']])
    spritzer_Qpdot_sgm = np.std(spritzer[['Qpdot']])
    spritzer_Qpdot_mu  = np.mean(spritzer[['Qpdot']])

    kalt_Qges_sgm      = np.std(kalt[['Qges']])
    kalt_Qges_mu       = np.mean(kalt[['Qges']])
    normal_Qges_sgm    = np.std(normal[['Qges']])
    normal_Qges_mu     = np.mean(normal[['Qges']])
    spritzer_Qges_sgm  = np.std(spritzer[['Qges']])
    spritzer_Qges_mu   = np.mean(spritzer[['Qges']])

    unspritzer         = data.loc[data['Zustand'] <= 1]
    spritzer           = data.loc[data['Zustand'] == 2]

    unsp_di_sgm        = np.std(unspritzer[['delta_I']])
    unsp_di_mu         = np.mean(unspritzer[['delta_I']])
    sp_di_sgm          = np.std(spritzer[['delta_I']])
    sp_di_mu           = np.mean(spritzer[['delta_I']])

    I    = np.arange(0, max_I+0.01, 0.01)
    Q    = np.arange(0, max_Q+0.01, 0.01)
    Qdot = np.arange(0, max_Qdot+0.01, 0.01)
    d_I  = np.arange(0, max_dI+0.01, 0.01)

    I_lo = fuzz.gauss2mf(I, min(I),1,kalt_I_mu[0], kalt_I_sgm[0])
    I_md = fuzz.gaussmf(I, normal_I_mu[0], normal_I_sgm[0])
    I_hi = fuzz.gauss2mf(I, spritzer_I_mu[0], spritzer_I_sgm[0], max(I), 1)

    Q_lo = fuzz.gauss2mf(Q, min(Q),1,kalt_Qges_mu[0], kalt_Qges_sgm[0])
    Q_md = fuzz.gaussmf(Q, normal_Qges_mu[0], normal_Qges_sgm[0])
    Q_hi = fuzz.gauss2mf(Q, spritzer_Qges_mu[0]+0.5, spritzer_Qges_sgm[0], max(Q), 1)

    Qdot_lo = fuzz.gauss2mf(Qdot, min(Qdot),1,kalt_Qpdot_mu[0], kalt_Qpdot_sgm[0])
    Qdot_md = fuzz.gaussmf(Qdot, normal_Qpdot_mu[0], normal_Qpdot_sgm[0])
    Qdot_hi = fuzz.gauss2mf(Qdot, spritzer_Qpdot_mu[0]+0.5, spritzer_Qpdot_sgm[0], max(Qdot),1)

    dI_lo = fuzz.gauss2mf(d_I, min(Q),1,unsp_di_mu[0], unsp_di_sgm[0])
    dI_hi = fuzz.gauss2mf(d_I, sp_di_mu[0], sp_di_sgm[0],max(Q),1)

    # print(spritzer_Qges_mu[0]+0.5,spritzer_Qpdot_mu[0]+0.5)

    # fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(8, 9))

    # ax0.plot(I, I_lo, 'b', linewidth=1.5, label='kalt')
    # ax0.plot(I, I_md, 'g', linewidth=1.5, label='normal')
    # ax0.plot(I, I_hi, 'r', linewidth=1.5, label='spritzer')
    # ax0.set_title('Strom')
    # ax0.legend()

    # ax1.plot(Q, Q_lo, 'b', linewidth=1.5, label='kalt')
    # ax1.plot(Q, Q_md, 'g', linewidth=1.5, label='normal')
    # ax1.plot(Q, Q_hi, 'r', linewidth=1.5, label='spritzer')
    # ax1.set_title('$Q_{ges}$')
    # ax1.legend()

    # ax2.plot(Qdot, Qdot_lo, 'b', linewidth=1.5, label='kalt')
    # ax2.plot(Qdot, Qdot_md, 'g', linewidth=1.5, label='normal')
    # ax2.plot(Qdot, Qdot_hi, 'r', linewidth=1.5, label='spritzer')
    # ax2.set_title('$\dot Q$')
    # ax2.legend()

    # ax3.plot(d_I, dI_lo, 'g', linewidth=1.5, label='kein spritzer')
    # ax3.plot(d_I, dI_hi, 'r', linewidth=1.5, label='spritzer')
    # ax3.set_title('$\Delta I$')
    # ax3.legend()

    # # Turn off top/right axes
    # for ax in (ax0, ax1, ax2, ax3):
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.get_xaxis().tick_bottom()
    #     ax.get_yaxis().tick_left()

    # plt.tight_layout()

    # plt.show()

    current   = strom
    waerme    = waerme
    Q_geschwi = Q_dot

    strom_level_lo  = fuzz.interp_membership(I, I_lo, current)
    strom_level_md  = fuzz.interp_membership(I, I_md, current)
    strom_level_hi  = fuzz.interp_membership(I, I_hi, current)
    waerme_level_lo = fuzz.interp_membership(Q, Q_lo, waerme)
    waerme_level_md = fuzz.interp_membership(Q, Q_md, waerme)
    waerme_level_hi = fuzz.interp_membership(Q, Q_hi, waerme)
    Qdot_level_lo   = fuzz.interp_membership(Qdot, Qdot_lo, Q_geschwi)
    Qdot_level_md   = fuzz.interp_membership(Qdot, Qdot_md, Q_geschwi)
    Qdot_level_hi   = fuzz.interp_membership(Qdot, Qdot_hi, Q_geschwi)

    regel1 = np.fmin(strom_level_hi, np.fmin(waerme_level_hi,Qdot_level_hi))
    dI_hi1 = np.fmin(regel1,dI_hi)

    regel2 = np.fmin(strom_level_md, np.fmin(waerme_level_hi,Qdot_level_hi))
    dI_hi2 = np.fmin(regel2,dI_hi)

    regel3 = np.fmin(strom_level_md, np.fmin(waerme_level_md,Qdot_level_md))
    dI_lo1 = np.fmin(regel3,dI_lo)

    regel4 = np.fmin(strom_level_lo, np.fmin(waerme_level_lo,Qdot_level_lo))
    dI_lo2 = np.fmin(regel4,dI_lo)

    regel5 = np.fmin(waerme_level_md, np.fmin(strom_level_lo,Qdot_level_lo))
    dI_lo3 = np.fmin(regel5,dI_lo)

    # k = 5
    # farben = cm.viridis(np.linspace(0,1,k))
    # fig, ax0 = plt.subplots(figsize=(8, 3))
    # d_I0 = np.zeros_like(d_I)
    # ax0.fill_between(d_I, d_I0, dI_hi1, facecolor=farben[0], alpha=0.7)
    # ax0.plot(d_I, dI_hi, color =farben[0], linewidth=1.5, linestyle='--', )
    # ax0.fill_between(d_I, d_I0, dI_hi2, facecolor=farben[1], alpha=0.7)
    # ax0.plot(d_I, dI_hi, color =farben[0], linewidth=1.5, linestyle='--')
    # ax0.fill_between(d_I, d_I0, dI_lo1, facecolor=farben[2], alpha=0.7)
    # ax0.plot(d_I, dI_lo, color =farben[-1], linewidth=1.5, linestyle='--')
    # ax0.fill_between(d_I, d_I0, dI_lo2, facecolor=farben[3], alpha=0.7)
    # ax0.plot(d_I, dI_lo, color =farben[-1], linewidth=1.5, linestyle='--')
    # ax0.fill_between(d_I, d_I0, dI_lo3, facecolor=farben[4], alpha=0.7)
    # ax0.plot(d_I, dI_lo, color =farben[-1], linewidth=1.5, linestyle='--')
    # ax0.set_title('Output membership activity')
    # plt.show()

    aggregated    = np.fmax(dI_lo3, np.fmax(np.fmax(dI_lo2, dI_lo1),np.fmax(dI_hi1,dI_hi2)))
    dI            = fuzz.defuzz(d_I, aggregated, 'centroid')
    dI_activation = fuzz.interp_membership(d_I, aggregated, dI)
    dI_h_percent = fuzz.interp_membership(d_I,dI_hi,dI)
    dI_l_percent = fuzz.interp_membership(d_I,dI_lo,dI)
    
    # fig, ax0 = plt.subplots(figsize=(8, 3))

    # ax0.plot(d_I, dI_hi, color = farben[0], linewidth=1.5, linestyle='--', )
    # ax0.plot(d_I, dI_lo, color = farben[-1], linewidth=1.5, linestyle='--')
    # ax0.fill_between(d_I, d_I0, aggregated, facecolor='gray', alpha=0.3)
    # ax0.plot([dI, dI], [0, dI_activation], 'k', linewidth=1.5, alpha=0.9)
    # ax0.set_title('Aggregated membership and result (line)')

    # # Turn off top/right axes
    # for ax in (ax0,):
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.get_xaxis().tick_bottom()
    #     ax.get_yaxis().tick_left()

    # plt.tight_layout()
    # plt.show()

    return dI,dI_h_percent,dI_l_percent


# dI, dI_h_percent, dI_l_percent = fuzzy_logic(
#                                             path = None, 
#                                             max_I = 13, 
#                                             max_Q = 15, 
#                                             max_Qdot = 25, 
#                                             max_dI = 0.6, 
#                                             strom = 9.2, 
#                                             waerme = 11.19 , 
#                                             Q_dot = 19.27
#                                             )
# print(dI, dI_h_percent, dI_l_percent)