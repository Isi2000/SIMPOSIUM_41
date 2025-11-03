# Higher-Order Singular Value Decomposition for Hydrogen Jet Combustion Analysis

## Overview

The focus of the work is now comparing HOSVD to PCA in combustion. The methodology decomposes multi-dimensional combustion data into orthogonal modes that capture spatial patterns, temporal evolution, and chemical species interactions. Briefly, the following steps have to be followed (checklist):
1.  Dowload a suitable dataset, possibly a DNS of hydrogen combustion (done)
2.  Define a preprocessing pipeline justitying all the choices made (almost done)
3.  Perform classical PCA on the reshaped tensor (x * y * time, number of chemical species) (almost done)
4.  Check the results of classical PCA with Parente 
5.  Perform HOSVD and if necessary HOOI (HOPCA)
6.  Check the results of HOSVD (HOOI/HOPCA)
7.  Compare the results quantitatively -> reconstruction errors
8.  Compare the resutls qualitatively -> do the PC mean something in both cases?
9.  Draw some conclusions
10. Run everything for real on CESVIMA
11. Write the paper (this needs to be done while the code runs)

## 1. Data Acquisition
Data is taken from BLASTNET at the following link:
https://blastnet.github.io/diluted_partially_premixed_h2air_lifted_flame

This is a Direct Numerical Simulation (DNS) data of hydrogen jet under multiple conditions, the ones currently analyzed are:

Each simulation provides:
- Spatial resolution: 1600 × 2000 grid points
- Temporal snapshots: 200 time steps
- Chemical species: 8 tracked species (H, H₂, O, O₂, OH, H₂O, HO₂, H₂O₂)

![Original Data at Time Step 10](README_PLOTS/original_data_t10.png)

## 2. Data Preprocessing

### Spatial Subsampling
In order to have the code run fast and test without losing too much time transfering everything on CESVIMA, the data is reduced heavily.

- Original grid: 1600 × 2000 points
- Subsampled grid: 200 × 160 points (10× reduction in each direction)

### Mass to Molar Fraction Conversion

Mass fractions are converted to molar fractions by dividing by species-specific molar masses, I wanna whatch the chemistry process, so I want moles. If I keep mass concentration the prevalent modes are not the species which are most involved in chemical reactions, but the heavier ones (learned this at my own expense :( )


### Data Processing

For the moment the only processing that I am making is logarithmic scaling to make sure that all variables are more on less on the same order of magnitude. I am not fully convinced about centering, usually PCA requires it, but I am working with quantities in the same domain and I don't want negative concentrations or masses (although in the log domain might be a good idea). 

This stuff is really important because the results are higly dependent on it. For the moment:
- log10 scale (ONLY!)

![Proessed Data at Time Step 10](README_PLOTS/processed_data_t10.png)


## 3. PCA (classical) (TO BE CHECKED!!)



