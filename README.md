# Higher-Order Singular Value Decomposition for Hydrogen Jet Combustion Analysis

## Overview

Sostanzialmente non ho tempo per fare la documentazione in due lingue e il paper, questa doc e' una sorta di placeholder raffazzonato su per sapere cosa faccio ogni volta che torno a lavorare sul progetto.

Inoltre lavorando in solitaria mi da una idea chiara della direzione che sta prendendo sto progetto (il fine ultimo), che detto francamente sta cambiando troppo troppo spesso per una roba che scade fra due giorni.

## 1. Data Acquisition
Data is taken from BLASTNET at the following link:
https://blastnet.github.io/diluted_partially_premixed_h2air_lifted_flame

This is a Direct Numerical Simulation (DNS) data of hydrogen jet under multiple conditions, the ones currently analyzed are:

Each simulation provides:
- Spatial resolution: 1600 × 2000 grid points
- Temporal snapshots: 200 time steps
- Chemical species: 8 tracked species (H, H₂, O, O₂, OH, H₂O, HO₂, H₂O₂)

## 2. Data Preprocessing

### Spatial Subsampling
In order to have the code run fast and test without losing too much time transfering everything on CESVIMA, the data is reduced heavily.

- Original grid: 1600 × 2000 points
- Subsampled grid: 200 × 160 points (10× reduction in each direction)

### Mass to Molar Fraction Conversion

A me interessa la chimica e se vedo qualcosa sicuramente non e' dovuto a come si muovono le masse, ma piuttosto a cosa reagisce con cosa. Dividento ciascuna specie per la massa molare ottengo la concentrazione.

### Data Processing

E' un bordello e la gente che si occupa di combustione e' un po' criptica e tratta la cosa in maniera  euristica, ossia ne prova qualcuna e va con metriche di ricostruzione. Questo approccio a me non piace tanto e ho buttato tempo a ragionare sulle trasformazioni.

#### Log

La logaritmica e' la piu' convincente di tutte. Le concentrazioni di specie chimiche vivono in uno spazio la cui piu' diretta rappresentazione e' quella logaritmica, sono a diversi ordini di grandezza e le cose non si sommano, ma bensi' moltilicano. Inoltre la trasformazione logaritmica e' supportata anche dal paper figo di Parente (e non solo), quindi sei proprio sicuro che vada bene.

Riassumendo: sul logaritmo siamo contenti, va bene

#### Mean and Standard scaling

Questo discorso e' decisamente piu' complicato e richiede un'analisi un po' piu' sottile. Per quando riguarda lo std scaling lascio la spiegazione che ho fatto per l'inglese che va bene.

For what concerns the division by variance the phsiscal interpretation of it requires a bit more care since after log scale the order of magnitude of the variables is the same.
It can be summarized in the following way:
- PCA on data not divided by variance -> components of maxixmum variabiliy in the teh dataset, I am finding the eigenvalues and eigenvectors of the matrix of covariance.
- PCA on data divided by variance -> components of maximux correlation. I am finding the components that correlates, regardless of its individual variability

Per il centering (mean scaling) sembrava una cagata, pero' a causa della forte struttura spaziotemporale della fiamma a H2 effettivamente importa se la media e' overall, temporale o spaziale. Spaziale fa si che il dataset sia meno suscettibile agli effetti di bordo della fiamma. Mi concentro sulle variazioni. Media totale e' l'approccio piu' agnostico, ma forse un po meno sensato, il grosso plus e' che la media totale non cambia proprio nulla nulla dei dati, quella temporale forse un pochetto si.

## 3. Classical PCA Analysis

### Methodology

Classical Principal Component Analysis (PCA) is performed by reshaping the 4D tensor (x, y, species, time) into a 2D matrix of shape (x·y·time, species). This treats each spatial-temporal point as an independent observation and identifies the linear combinations of chemical species that capture the most variance in the data.

The preprocessing applied before PCA includes:
- Log₁₀ scaling to handle the large dynamic range of species concentrations
- Epsilon floor of 1e-12 to avoid logarithm singularities

Singular Value Decomposition (SVD) is used to compute the principal components:
```
X = U Σ Vᵀ
```
where:
- **U** contains the spatial-temporal patterns (Φ, spatial modes)
- **Σ** contains the singular values (related to variance explained)
- **V** contains the loadings (contribution of each species to each mode)


## 5. HOSVD (Higher-Order Singular Value Decomposition)

### Methodology

Unlike classical PCA which reshapes the tensor into a 2D matrix, HOSVD preserves the multi-dimensional structure of the combustion data. The 4D tensor (x, y, species, time) is decomposed using Tucker decomposition:

```
X ≈ G ×₁ U₁ ×₂ U₂ ×₃ U₃ ×₄ U₄
```

where:
- **G** is the core tensor containing the interaction between all modes
- **U₁, U₂** are the spatial factor matrices (x and y dimensions)
- **U₃** is the chemical species factor matrix (U_chem in code)
- **U₄** is the temporal factor matrix (U_time)

The decomposition is computed by:
1. Performing SVD on each mode unfolding of the tensor
2. Computing the core tensor via multi-mode dot product (tensorly)
3. The core tensor G captures the interactions between spatial patterns, chemical species, and temporal evolution

### Reconstruction Accuracy

The HOSVD decomposition achieves excellent reconstruction accuracy:
- **Relative reconstruction error**: 2.00e-15 (stability)

### Core Tensor Singular Values

The core tensor singular values reveal the importance of each mode across different dimensions:

![HOSVD Core Tensor Singular Values ](README_PLOTS/hosvd_core_vals.png)
**CORE ** HOSVD core tensor singular values across spatial, chemical, and temporal modes.

- **Spatial Dimensions (X and Y)**: Show rapid decay, indicating that spatial patterns can be captured with relatively few modes
- **Chemical Dimension**: This is the term of comparison (qua c'e' la ciccia, ma e' tardi e lo faccio domani)
- **Time Dimension**: Shows distinct temporal mode importance, with the first few modes capturing most of the temporal dynamics


## Matematica

Le dimostrazioni matematiche di equivalenza sono in latex, non c'e' dubbio che mi metta ora a cosi' poco dalla scandenza a scriverle anche in md.

## Considerazione sui due metodi

In sostanza se mi metto a guardare una compoennte per volta ottengo che HOSVD e' esattamente uguale a PCA. Questa cosa non fa tanto piacere perche' toglie importanza all'algoritmo, pero' guardando il lato positivo significa anche che dato che PCA e' contenuta dentro HOSVD e gli algoritmi sono computazionalmente risibili HOSVD e' strettamente migliore di PCA.

La differenza sostanziale fra i due algoritmi sta nel core tensor. Per construzione gli elementi del core tensor cattura interazioni fra PC calcolate lungo diversi assi del tensore. 
