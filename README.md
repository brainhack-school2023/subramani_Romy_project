
`Generated_data` handles all the intermediately generated files which are necessary for the subsequent analysis. `Results` : to store all the figures; `src_data` : all the necessary source data files; `src_scripts` : scripts for the whole pipeline

`src_scripts` contains the scripts in a sequential order required for the analysis.

A quick description : 

`1parsing_for_subject.py` : followed the work from Nentwich and co. https://doi.org/10.1016/j.neuroimage.2020.117001, we identified the subjects based on several criteria)

`2Downloading_from_AWS.sh` : with the identified subjects, next is to fetch the Healthy Brain Network (HBN) dataset from their AWS bucket. More info: http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/ 

`3loading_datasets.py` : Once the datasets are pulled in, next is to unload them, some preprocessing and export the files to `Generated_data`

`4Source_Inversion.py` : EEG signal from Scalp space to Brain surface AKA Source Localization / Source Inversion implemented using MNE, mapping to Glasser et al. atlas, applying Hilbert transform, and then bandpassing the signal into Theta, Alpha, Low and High Beta

`5ISC_&_bootstrapping.py` : and its dependent cousin `util_5_CorrCCA.py` The latter deals with estimating correlation of EEG cortical signal, pythoned by https://github.com/pa-ak/ISC-Inter-Subject-Correlations, is introduced by Dmochowski et al. 2012 https://doi.org/10.3389%2Ffnhum.2012.00112; the former creates null surrogate distribution for statistical comparisons

`6_Baseline_correction.py` : zscoring the EEG cortical activity

`7a_SDI_computation_differenced.py` : Structural-Decoupling Index (SDI) computed on the zscored EEG activity during Strong ISC. Both empirical and surrogate SDIs are computed

`7b_SDI_computation_baseline.py` : SDI computed for the (absolute) Baseline. Both empirical and surrogate SDIs are computed

`8_SDI_statistics.py` : 2-level-model for statistical comparison between empirical SDI and Surrogate SDI

`9_SDI_spatial_maps.py` : Visualization of the spatial maps

`10_SDI_in_7YeoNetworks.py` : SDI analysis in network-level in the 7-network variant of Yeo networks

`10_SDI_in_17YeoNetworks.py` : SDI analysis in a finer network level, i.e., 17-network variant of Yeo networks

`11_SDI_Decoding.py` : Meta-analysis of the spatial maps, and visualization in Wordclouds

