## Study 1
`Generated_data` handles all the intermediately generated files which are necessary for the subsequent analysis. `Results` : to store all the figures; `src_data` : all the necessary source data files; `src_scripts` : scripts for the whole pipeline

`src_scripts` contains the scripts in a sequential order required for the analysis.

A quick description : 
`1_weak_ISC_definition.py` : Will contain script related to defining periods of Weak ISC and the EEG data for those periods are sliced.

`2_Baseline_correction.py` : zscoring the EEG cortical activity

`3a_SDI_computation_differenced.py` : Structural-Decoupling Index (SDI) computed on the zscored EEG activity during Strong ISC. Both empirical and surrogate SDIs are computed

`3b_SDI_computation_baseline.py` : SDI computed for the (absolute) Baseline. Both empirical and surrogate SDIs are computed

`4_SDI_statistics.py` : 2-level-model for statistical comparison between empirical SDI and Surrogate SDI

`5_SDI_spatial_maps.py` : Visualization of the spatial maps

`6_SDI_statistics_strong_weak_comparison.py` : Statistical comparisons between SDIs computed for Weak and Strong ISC periods

## Study 2
`7_Segregation.py` : Interactions between a given Yeo-Krienen network vs the rest of the network using a Graph Theory measure. Subsequent analysis on the interaction during certain segments of the video