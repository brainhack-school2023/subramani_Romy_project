
#%%
import numpy as np

HOMEDIR = "/Users/venkatesh/Desktop/BHS/subramani_project/"

weak_ISC = [ 10,  32,  35,  46,  47,  58,  59,  60,  61,  70,  71,  94,  95,
       107, 108, 123, 124, 132, 137, 142, 146, 148, 155, 156, 158]

fs = 125
_200ms_in_samples = 25
_500ms_in_samples = 63
n_sub = 25

def slicing(stc):
    signal_sliced = list()

    for time in weak_ISC:
        sliced = stc[:, :,time * fs - _200ms_in_samples : time * fs + _500ms_in_samples]
        signal_sliced.append(sliced)
    


    return signal_sliced


envelope_signal = np.load("/Users/venkatesh/Desktop/BHS/subramani_project/src_data/envelope_signal_bandpassed.npz")

envelope_signal_sliced = dict()
for labels, signal in envelope_signal.items():
    envelope_signal_sliced[f'{labels}'] = slicing(signal)


np.savez_compressed(f'{HOMEDIR}//Generated_data/Cortical_surface_related/wideband_and_other_bands',**envelope_signal_sliced)
# %%
