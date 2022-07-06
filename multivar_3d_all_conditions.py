# -*- coding: utf-8 -*-

import os
import sys
from contextlib import contextmanager
from tqdm import tqdm
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt 
import pymc3 as pm
import arviz as az
import theano.tensor as tt
import pickle


path = "D:/Git/Reading/erp_gaussian_group_condition/"

#####plotting parameters
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'figure.titlesize': 14})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"

############################## Import and prepare epochs ######################
os.chdir(path+'results/')

ampl = np.load(path+"data_l/leaerners_200ms_baseline.npy")
ampnl = np.load(path+"data_nl/non_leaerners_200ms_baseline.npy")

amps = np.array([ampl,ampnl])

times = np.load(path+"data_nl/times_200ms_baseline.npy")

chans = pd.read_csv(path+"/data_l/chans.csv")['chans'].values

G = amps.shape[0] #numver of groups G
C = amps.shape[1] #number of conditions C
E = amps.shape[2] #number of electrodes E 
S = amps.shape[3] #number of time-samples S

ts = np.arange(S)/256


########### Build model ############

with pm.Model() as mod: #Multivariate Normal model with LKJ prior
    sd = pm.HalfNormal.dist(1.0)
    chol, corr, std = pm.LKJCholeskyCov("chol", n=E, eta=6.0, sd_dist=sd, compute_corr=True)
    cov = pm.Deterministic("cov", chol.dot(chol.T))
    m = pm.Normal('m', mu=0, sigma=1.0)
    g = pm.Normal("g", mu=m, shape=(S,E,C,G))
    M = pm.Deterministic("M", tt.dot(cov, g.T)).swapaxes(0,2).swapaxes(0,1)
    e = pm.HalfNormal("e", 5.0)+1.0
    y = pm.Normal("y", mu=M, sigma=e, observed=amps, shape=amps.shape)

########### Prio Predictive Checks ################    
#amps = amps.reshape(G,C,E,S)

with mod:
    ppc = pm.sample_prior_predictive(samples=2000, var_names=["y"])

ppc = ppc['y']
#ppc = np.swapaxes(ppc, 1,2)
#amps = amps.swapaxes(0,1)
hdis = az.hdi(ppc[:,0,0,12,:], hdi_prob=0.9)
plt.plot(times, amps[0,0,12,:], color='grey', linestyle='-', label='Learners: observed mean')
plt.plot(times, ppc[:,0,0,12,:].mean(axis=0), color='m', linestyle='-', label='Learners: predicted mean')
plt.fill_between(times, hdis[:,0],hdis[:,1], color='m', alpha=0.3)
hdis = az.hdi(ppc[:,1,0,12,:], hdi_prob=0.9)
plt.plot(times, amps[1,0,12,:], color='grey', linestyle='--', label='Non-learners: observed mean')
plt.plot(times, ppc[:,1,0,12,:].mean(axis=0), color='c', linestyle='--', label='Non-learners: predicted mean')
plt.fill_between(times, hdis[:,0],hdis[:,1], color='c', alpha=0.3)
plt.axvline(0, color='k', linestyle=':', zorder=1, alpha=0.5)
plt.axhline(0, color='k', linestyle=':', zorder=1, alpha=0.5)
plt.title('Prior Predictive Checks (Pz)')
plt.ylabel('Amplitude (μV)')
plt.xlabel('Time (s)')
plt.legend()
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.savefig('ppcs_pz.png', dpi=300)
plt.close()


# ###################### SAMPLE ##########################    
with mod:
    trace = pm.sample(1000, tune=1000, chains=4, cores=8, init='adapt_diag',
                      compute_convergence_checks=False, target_accept=0.9)


# ######################## save trace ###########################
tracedir = path+"trace/"
pm.backends.ndarray.save_trace(trace, directory=tracedir, overwrite=True)

##load trace
# with mod:
#     trace = pm.load_trace(tracedir)


######################## Plot Example electrode #################
#os.chdir("D:/_0Reading/_OHBM/eeg_grw/grw_3d_groups/posterior_estimates_comparison/")
tnames = ['Tone 4 (target)', 'Tone 1', 'Tone 2', 'Tone 3']
fnames = ['Tone4', 'Tone1', 'Tone2', 'Tone3']
post = trace['M']#.reshape(trace['M'].shape[0], G, C, E, S)
for c in range(C):
    hdis = az.hdi(post[:,12,0,c,:], hdi_prob=0.9)
    plt.plot(times, amps[0,c,12,:], color='grey', linestyle='-', label='Learners: observed mean')
    plt.plot(times, post[:,12,0,c,:].mean(axis=0), color='m', linestyle='-', label='Learners: posterior mean')
    plt.fill_between(times, hdis[:,0],hdis[:,1], color='m', alpha=0.3)
    hdis = az.hdi(post[:,12,1,c,:], hdi_prob=0.9)
    plt.plot(times, amps[1,c,12,:], color='grey', linestyle='--', label='Non-learners: observed mean')
    plt.plot(times, post[:,12,1,c,:].mean(axis=0), color='c', linestyle='--', label='Non-learners: posterior mean')
    plt.fill_between(times, hdis[:,0],hdis[:,1], color='c', alpha=0.3)
    plt.axvline(0, color='k', linestyle=':', zorder=1, alpha=0.5)
    plt.axhline(0, color='k', linestyle=':', zorder=1, alpha=0.5)
    plt.title('Pz '+tnames[c])
    plt.ylabel('Amplitude (μV)')
    plt.xlabel('Time (s)')
    plt.legend(fontsize=12)
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.tight_layout()
    plt.savefig(fnames[c]+'_posteriors.png', dpi=300)
    plt.close()

############ Plot topomaps ###############

info_path = path+"data_l/info_200ms_baseline.pickle"
with open(info_path, 'rb') as handle:
    info = pickle.load(handle)
    
#save and plot correlations
corr = trace['chol_corr'].mean(axis=0)
corr = pd.DataFrame(corr, columns=chans, index=chans)
corr.to_csv("electrodes_correlations.csv")

plt.rcParams['text.color'] = "white"
plt.rcParams.update({'font.size': 16})
cortopo = mne.viz.plot_topomap(corr['Pz'].values, info, names=chans, 
                               show_names=True, cmap='viridis', show=False)
plt.rcParams['text.color'] = "k"
plt.rcParams.update({'font.size': 16})
cbar = plt.colorbar(cortopo[0])
cbar.set_label('correlation to Pz')
plt.tight_layout()
plt.savefig('corrs_topo.png', dpi=300)
plt.close()


# ##plot learners
# fig = plt.figure()
# h5 = [az.hdi(post.mean(axis=0)[c,0,0,90:192], hdi_prob=0.9)[0] for c in range(len(chans))]
# mean = post.mean(axis=0)[:,0,0,90:192].mean(axis=1)
# h95 = [az.hdi(post.mean(axis=0)[c,0,0,90:192], hdi_prob=0.9)[1] for c in range(len(chans))]
# ax1 = fig.add_axes([0,1,1,1]) 
# ax1 = mne.viz.plot_topomap(h5, epo.info, names=chans, vmax=3, vmin=-3, show=False)[0]
# plt.title('HDI 5%', fontsize=22)
# ax2 = fig.add_axes([1,1,1,1]) 
# ax2 = mne.viz.plot_topomap(mean, epo.info, names=chans, vmax=3, vmin=-3, show=False)[0]
# plt.title('Posterior Mean', fontsize=22)
# ax3 = fig.add_axes([2,1,1,1]) 
# ax3 = mne.viz.plot_topomap(h95, epo.info, names=chans, vmax=3, vmin=-3, show=False)[0]
# plt.title('HDI 95%', fontsize=22)
# ax4 = fig.add_axes([1,2,1,0.2]) 
# ax4.axis('off')
# plt.title('Tone 4 Learners Electrodes Estimates (350-750ms average)', fontsize=30)
# ax5 = fig.add_axes([3,1,0.1,1]) 
# ax5.axis('off')
# cbar = plt.colorbar(ax2, fraction=1, pad=1)
# cbar.set_label('μV')
# plt.savefig('learners_topo_tone4.png', dpi=300)

    

# ##plot non-learners
# h5 = [az.hdi(post.mean(axis=0)[c,1,0,90:192], hdi_prob=0.9)[0] for c in range(len(chans))]
# mean = post.mean(axis=0)[:,1,0,90:192].mean(axis=1)
# h95 = [az.hdi(post.mean(axis=0)[c,1,0,90:192], hdi_prob=0.9)[1] for c in range(len(chans))]
# fig = plt.figure()
# ax1 = fig.add_axes([0,1,1,1]) 
# ax1 = mne.viz.plot_topomap(h5, epo.info, names=chans, vmax=3, vmin=-3, show=False)[0]
# plt.title('HDI 5%', fontsize=22)
# ax2 = fig.add_axes([1,1,1,1]) 
# ax2 = mne.viz.plot_topomap(mean, epo.info, names=chans, vmax=3, vmin=-3, show=False)[0]
# plt.title('Posterior Mean', fontsize=22)
# ax3 = fig.add_axes([2,1,1,1]) 
# ax3 = mne.viz.plot_topomap(h95, epo.info, names=chans, vmax=3, vmin=-3, show=False)[0]
# plt.title('HDI 95%', fontsize=22)
# ax4 = fig.add_axes([1,2,1,0.2]) 
# ax4.axis('off')
# ax4.set_title('Tone 4 Non-learners Electrodes Estimates (350-750ms average)', fontsize=30)
# ax5 = fig.add_axes([3,1,0.1,1]) 
# ax5.axis('off')
# cbar = plt.colorbar(ax2, fraction=1, pad=1)
# cbar.set_label('μV')
# fig.savefig('non-learners_topo_tone4.png', dpi=300)


########## Posterior predictive 
with mod:
    preds = pm.sample_posterior_predictive(trace)
tnames = ['Tone 4 (target)', 'Tone 1', 'Tone 2', 'Tone 3']
fnames = ['Tone4', 'Tone1', 'Tone2', 'Tone3']
pred = preds['y']
pred = pred.reshape(pred.shape[0],G,C,E,S)
for c in range(C):
    hdis = az.hdi(pred[:,0,c,12,:], hdi_prob=0.9)
    plt.plot(times, amps[0,c,12,:], color='grey', linestyle='-', label='Learners: observed mean')
    plt.plot(times, pred[:,0,c,12,:].mean(axis=0), color='m', linestyle='-', label='Learners: posterior mean')
    plt.fill_between(times, hdis[:,0],hdis[:,1], color='m', alpha=0.3)
    hdis = az.hdi(pred[:,1,c,12,:], hdi_prob=0.9)
    plt.plot(times, amps[1,c,12,:], color='grey', linestyle='--', label='Non-learners: observed mean')
    plt.plot(times, pred[:,1,c,12,:].mean(axis=0), color='c', linestyle='--', label='Non-learners: posterior mean')
    plt.fill_between(times, hdis[:,0],hdis[:,1], color='c', alpha=0.3)
    plt.axvline(0, color='k', linestyle=':', zorder=1, alpha=0.5)
    plt.axhline(0, color='k', linestyle=':', zorder=1, alpha=0.5)
    plt.title('Pz '+tnames[c])
    plt.ylabel('Amplitude (μV)')
    plt.xlabel('Time (s)')
    plt.legend(fontsize=12)
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.savefig(fnames[c]+'_post_predicted.png', dpi=300)
    plt.close()


# ##plot learners topo predicted
# fig = plt.figure()
# h5 = [az.hdi(post.mean(axis=0)[0,0,c,90:192], hdi_prob=0.9)[0] for c in range(len(chans))]
# mean = post.mean(axis=0)[0,0,:,90:192].mean(axis=1)
# h95 = [az.hdi(post.mean(axis=0)[0,0,c,90:192], hdi_prob=0.9)[1] for c in range(len(chans))]
# ax1 = fig.add_axes([0,1,1,1]) 
# ax1 = mne.viz.plot_topomap(h5, epo.info, names=chans, vmax=3, vmin=-3, show=False)[0]
# plt.title('HDI 5%', fontsize=22)
# ax2 = fig.add_axes([1,1,1,1]) 
# ax2 = mne.viz.plot_topomap(mean, epo.info, names=chans, vmax=3, vmin=-3, show=False)[0]
# plt.title('Posterior Mean', fontsize=22)
# ax3 = fig.add_axes([2,1,1,1]) 
# ax3 = mne.viz.plot_topomap(h95, epo.info, names=chans, vmax=3, vmin=-3, show=False)[0]
# plt.title('HDI 95%', fontsize=22)
# ax4 = fig.add_axes([1,2,1,0.2]) 
# ax4.axis('off')
# plt.title('Tone 4 Learners Electrodes Estimates (350-750ms average)', fontsize=30)
# ax5 = fig.add_axes([3,1,0.1,1]) 
# ax5.axis('off')
# cbar = plt.colorbar(ax2, fraction=1, pad=1)
# cbar.set_label('μV')
# plt.savefig('learners_topo_tone4_predicted.png', dpi=300)

    

# ##plot non-learners topo predicted
# h5 = [az.hdi(post.mean(axis=0)[1,0,c,90:192], hdi_prob=0.9)[0] for c in range(len(chans))]
# mean = post.mean(axis=0)[1,0,:,90:192].mean(axis=1)
# h95 = [az.hdi(post.mean(axis=0)[1,0,c,90:192], hdi_prob=0.9)[1] for c in range(len(chans))]
# fig = plt.figure()
# ax1 = fig.add_axes([0,1,1,1]) 
# ax1 = mne.viz.plot_topomap(h5, epo.info, names=chans, vmax=3, vmin=-3, show=False)[0]
# plt.title('HDI 5%', fontsize=22)
# ax2 = fig.add_axes([1,1,1,1]) 
# ax2 = mne.viz.plot_topomap(mean, epo.info, names=chans, vmax=3, vmin=-3, show=False)[0]
# plt.title('Posterior Mean', fontsize=22)
# ax3 = fig.add_axes([2,1,1,1]) 
# ax3 = mne.viz.plot_topomap(h95, epo.info, names=chans, vmax=3, vmin=-3, show=False)[0]
# plt.title('HDI 95%', fontsize=22)
# ax4 = fig.add_axes([1,2,1,0.2]) 
# ax4.axis('off')
# ax4.set_title('Tone 4 Non-learners Electrodes Estimates (350-750ms average)', fontsize=30)
# ax5 = fig.add_axes([3,1,0.1,1]) 
# ax5.axis('off')
# cbar = plt.colorbar(ax2, fraction=1, pad=1)
# cbar.set_label('μV')
# fig.savefig('non-learners_topo_tone4_predicted.png', dpi=300)


############### plot contrasts ########################

ampsl = amps[0,0,:,:] - amps[0,1:4,:,:].mean(axis=0)
ampsnl = amps[1,0,:,:] - amps[1,1:4,:,:].mean(axis=0)

#posterior
postl = post[:,12,0,0,:] - post[:,12,0,1:4,:].mean(axis=1)
postnl = post[:,12,1,0,:] - post[:,12,1,1:4,:].mean(axis=1)
postp = postl - postnl #post[:,0,:,:] - post[:,1,:,:]
hdis = az.hdi(postp, hdi_prob=0.9)
plt.plot(times, ampsl[12,:]-ampsnl[12,:], color='grey', linestyle='-', label='Learners - Non-learners (observed)')
plt.plot(times, postp.mean(axis=0), color='mediumvioletred', linestyle='-', label='Learners - Non-learners (posterior)')
plt.fill_between(times, hdis[:,0],hdis[:,1], color='mediumvioletred', alpha=0.2, label='Predicted 90% HDI')
plt.axvline(0, color='k', linestyle=':', zorder=1, alpha=0.5)
plt.axhline(0, color='k', linestyle=':', zorder=1, alpha=0.5)
plt.title('Pz')
plt.ylabel('Amplitude (μV)')
plt.xlabel('Time (s)')
plt.legend(fontsize=10, loc='lower right')
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.savefig('posterior_target_non-target_learner_nonlear_difference.png', dpi=300)
plt.close()

#predicted
predl = pred[:,0,0,12,:] - pred[:,0,1:4,12,:].mean(axis=1)
prednl = pred[:,1,0,12,:] - pred[:,1,1:4,12,:].mean(axis=1)
predp = predl - prednl
me = predp.mean(axis=0)
sd = predp.std(axis=0)
sds = np.array([me-sd, me+sd])
plt.plot(times, ampsl[12,:]-ampsnl[12,:], color='grey', linestyle='-', label='Learners - Non-learners (observed)')
plt.plot(times, me, color='darkviolet', linestyle='-', label='Learners - Non-learners (predicted)')
plt.fill_between(times, sds[0,:], sds[1,:], color='darkviolet', alpha=0.2, label='Predicted SD')
plt.axvline(0, color='k', linestyle=':', zorder=1, alpha=0.5)
plt.axhline(0, color='k', linestyle=':', zorder=1, alpha=0.5)
plt.title('Pz')
plt.ylabel('Amplitude (μV)')
plt.xlabel('Time (s)')
plt.legend(fontsize=10, loc='lower right')
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.savefig('ppc_learner_nonlear_difference_sd.png', dpi=300)
plt.close()




######### Save summaries ##########
summ = az.summary(trace, hdi_prob=0.9, round_to=4)
summ = pd.DataFrame(summ)
summ.to_csv('summary.csv')
print("summary saved")

bfmi = az.bfmi(trace)
bfmi = pd.DataFrame(bfmi)
bfmi.to_csv('bfmi.csv')
print("bfmi saved") 

ener = az.plot_energy(trace)
plt.savefig("energy.png", dpi=300)
plt.close()

########### Model fit

loo = az.loo(trace, pointwise=True)
loo = pd.DataFrame(loo)
loo.to_csv("loo.csv")

waic = az.waic(trace, pointwise=True)
waic = pd.DataFrame(waic)
waic.to_csv('waic.csv')

###plot rank
pathtranks = path+"tranks/"
varias = trace.varnames[1:3]+trace.varnames[4:]
for var in tqdm(varias):
    err = az.plot_rank(trace, var_names=[var], kind='vlines', ref_line=True,
                       vlines_kwargs={'lw':1}, marker_vlines_kwargs={'lw':2})
    plt.savefig(pathtranks+var+'_trank.png', dpi=300)
    plt.close()









