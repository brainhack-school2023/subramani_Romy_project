#%%
import numpy as np
import mne_connectivity
from nilearn import datasets, plotting, maskers
from collections import defaultdict
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed

import multiprocessing.pool


HOMEDIR = '/users/local/Venkatesh/Brainhack'
n_subjects = 25

atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011()
yeo = atlas_yeo_2011.thick_7
glasser = "/users2/local/Venkatesh/Multimodal_ISC_Graph_study/src_data/Glasser_masker.nii.gz"  # f"{HOMEDIR}/src_data/Glasser_masker.nii.gz"

masker = maskers.NiftiMasker(standardize=False, detrend=False)
masker.fit(glasser)
glasser_vec = masker.transform(glasser)

yeo_vec = masker.transform(yeo)
yeo_vec = np.round(yeo_vec)

matches = []
match = []
best_overlap = []
for i, roi in enumerate(np.unique(glasser_vec)):
    overlap = []
    for roi2 in np.unique(yeo_vec):
        overlap.append(
            np.sum(yeo_vec[glasser_vec == roi] == roi2) / np.sum(glasser_vec == roi)
        )
    best_overlap.append(np.max(overlap))
    match.append(np.argmax(overlap))
    matches.append((i + 1, np.argmax(overlap)))

n_seconds = 170
fs = 125
n_ROI = 360

envelope_signal = np.load(f'{HOMEDIR}/src_data/envelope_signal_bandpassed.npz')
envelope_signal_alpha = envelope_signal['alpha']

#%%
def segregation_computation(envelope, network):
    window_size = 125 #1s

    sce_whole_size = list()
    
    def window_parallel(window_start):
        signal = envelope[:, :, window_start:window_start+window_size]
        sce = mne_connectivity.spectral_connectivity_epochs(signal, sfreq = 125, fmin=8, fmax =13, method='wpli', faverage=True, verbose=False)
        
        connectivity_matrix = sce.get_data().reshape(n_ROI, n_ROI)
        connectivity_matrix_symmetric = connectivity_matrix +connectivity_matrix.T
        print(window_start)
        return connectivity_matrix_symmetric
    

    pool = multiprocessing.pool.ThreadPool(170)
    sce_whole_size = pool.map(window_parallel, range(170))


    segregation_list = list()

    for window in (range(len(sce_whole_size))):
    
        network_indices = [np.where(np.array(match)==network)[0]][0]
        
        binmask_within = np.zeros((n_ROI, n_ROI))
        binmask_between_step1 = np.zeros((n_ROI, n_ROI))

        for indices in network_indices:
            

            binmask_within[indices, network_indices]=1
            binmask_within[network_indices, 1]=1

            binmask_between_step1[indices, :] = 1
            binmask_between_step1[:, indices] = 1

            binmask_between = binmask_between_step1-binmask_within
        
        matrix = sce_whole_size[window]
        
        average_strength_within = np.sum(matrix * binmask_within)/ (np.sum(binmask_within) - len(network_indices))
        
        average_strength_between = np.sum(matrix * binmask_between)/np.sum(binmask_between)

        segregation = (average_strength_within - average_strength_between)/average_strength_within
        
        segregation_list.append(segregation)

    return segregation_list

# %%


NB_CPU = multiprocessing.cpu_count()

def parellelize(runs):
    np.random.seed(runs)
    return segregation_computation(envelope_signal_alpha[:, np.random.permutation(n_ROI), :], 1)
    

surrogate_data = Parallel(n_jobs=NB_CPU - 5, max_nbytes=None)(delayed(parellelize)(main_parallelization) for main_parallelization in tqdm(range(100)))

# %%
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.figure(figsize=(25,10))

# %%
plt.plot(np.array(surrogate_data).T,c='r', label='surrogate')
# plt.plot(segregation_dict['1'],c='b',label='true')
plt.legend()
# %%

# %%
np.shape(surrogate_data)
# %%
s = segregation_computation(envelope_signal_alpha, 1)
# %%
import time
s = time.time()
# sce= mne_connectivity.spectral_connectivity_epochs(envelope_signal_alpha[:,:,125:250], sfreq = 125, fmin=8, fmax =13, method='wpli', faverage=True)

# connectivity_matrix = sce.get_data().reshape(n_ROI, n_ROI)
# connectivity_matrix_symmetric = connectivity_matrix +connectivity_matrix.T

for window in (range(269)):

    network_indices = [np.where(np.array(match)==1)[0]][0]
    
    binmask_within = np.zeros((n_ROI, n_ROI))
    binmask_between_step1 = np.zeros((n_ROI, n_ROI))

    for indices in network_indices:
        

        binmask_within[indices, network_indices]=1
        binmask_within[network_indices, 1]=1

        binmask_between_step1[indices, :] = 1
        binmask_between_step1[:, indices] = 1

        binmask_between = binmask_between_step1-binmask_within        
#%%
import time
s = time.time()
segregation_computation(envelope_signal_alpha[:, np.random.permutation(n_ROI), :], 1)
time.time() - s

# %%



# %%
plt.plot(parallelized_segregation)
# %%

# %%
