# The PyClustKit Module: All about clustering in a single Python Module!

The pyclustkit module is built to include various state-of-the-art meta-features for algorithm selection in clustering 
and many clustering validity indices (CVIs) with optimizations to enhance computational speed when calculating many. 

---
### ⚠️ WARNING: For the Demo Version refer [here](https://github.com/automl-uprc/PyClust-Demo)⚠️

---

## Table of Contents
- [Requirements](#requirements)
- [Installation Instructions](#installation-instructions)
- [Useful Links](#useful-links)
- [Usage Examples](#usage-examples)
- [GUI Version](#gui-version)
- [Cite Us!](#citing-this-work)

# Requirements

This version of pyclustkit requires 
- Python>=3.12
- dgl==2.2.1 
- torch==2.3.0
- numpy<2.0

# Installation Instructions

The pyclustkit is available to download with pypi

```commandline
pip install pyclustkit
```

 ⚠️ **Warning** : Required version of dgl==2.2.1 may not be found in the pypi index when using Ubuntu/Debian Releases. 
This is an issue to be fixed in future versions of the library. For now you can use 

```commandline
pip install pyclustkit -f https://data.dgl.ai/wheels/torch-2.3/repo.html
```


# Usage Examples
## Calculating Internal Cluster Validity Indices (CVI) 

PyClustKit comes with an evaluation suite of 46 internal validity indices. Each is implemented on top of numpy and, 
the module incorporates specific methods for speeding up the execution of multiple CVI by implementing a shared process 
tracking. 


```{python}
from pyclustkit.eval import CVIToolbox 

ct = CVIToolbox(X,y)
ct.calculate_icvi(cvi=["dunn", "silhouette"]) # if no CVI are specified it defaults to 'all'.
print(ct.cvi_results)

```
## Meta Learning 

### Meta-Feature Extraction
PyClustKit comes with an evaluation suite of 46 internal validity indices. Each is implemented on top of numpy and, 
the module incorporates specific methods for speeding up the execution of multiple CVI by implementing a shared process 
tracking. 


```{python}
from pyclustkit.eval import CVIToolbox 

ct = CVIToolbox(X,y)
ct.calculate_icvi(cvi=["dunn", "silhouette"]) # if no CVI are specified it defaults to 'all'.
print(ct.cvi_results)

```

## GUI Version 
Users can also opt to use the demo version of the library which comes with a graphical interface based on Gradio  and 
various utilities such as data generation, meta-learner training and visuals. 

### (A)  Ready-To-Use Docker Image

```commandline
docker run -p 7861:7861 giannispoy/pyclust 
```
### (B) Demo Git Repository 

Users can also download the code on the [pyclust-demo repository](https://github.com/automl-uprc/PyClust-Demo) and follow the instructions in the corresponding README 
file.


### Implemented CVIs

<details>
<summary>List of Implemented CVI with citations</summary>
Currently the collection consists of the following internal CVIs. R does not do gdi 61,62,63 due to hausdorff:

1. **ball_hall**: <i> G. H. Ball and D. J. Hall. Isodata: A novel method of data analysis and pattern
                      classification. Menlo Park: Stanford Research Institute. (NTIS No. AD 699616),1965.</i>
2. **banfeld_raftery**: <i> J.D. Banfield and A.E. Raftery. Model-based gaussian and non-gaussian clustering. Biometrics,
                        49:803–821, 1993. </i>
3. **c_index**: <i> Hubert, Lawrence & Levin, Joel. (1976). A general statistical framework for assessing categorical 
clustering in free recall. Psychological Bulletin. 83. 1072-1080. 10.1037/0033-2909.83.6.1072. </i>
4. **CDbw** : <i>Halkidi, M., & Vazirgiannis, M. (2008). A density-based cluster validity approach using 
multi-representatives. Pattern Recognit. Lett., 29, 773-786.  </i>
5. **det_ratio** : <i> A. J. Scott and M. J. Symons. Clustering methods based on likelihood ratio criteria. Biometrics, 
                27:387–397, 1971.</i>
6. **Dunn Index** : <i>J. Dunn. Well separated clusters and optimal fuzzy partitions. Journal of Cybernetics, 4:95–104, 
                    1974. </i>

7. **GDI [11,21,31,41,51,61][12,22,32,42,52,62][13,23,33,43,53,63]**: <i>J. C. Bezdek and N. R. Pal. Some new indexes of
cluster validity. IEEE Transactions on Systems, Man, and CyberneticsÑPART B: CYBERNETICS, 28, no.3:301–315, 1998.</i>
8. **ksq_detw**:  F. H. B. Marriot. Practical problems in a method of cluster analysis. Biometrics,
27:456–460, 1975.
9. **log_det_ratio**: <i> Halkidi et al. On clustering validation techniques. J. Intell. Inf. Syst., 2001. </i>
10. **log_ss_ratio**: <i> J. A. Hartigan. Clustering algorithms. New York: Wiley, 1975. </i>
11. **McClain_Rao**: <i> J. O. McClain and V. R. Rao. Clustisz: A program to test for the quality of
                         clustering of a set of objects. Journal of Marketing Research, 12:456–460, 1975.</i>

11. trace_w Index

13. Friedman-Rudin 1 Index
14. Friedman-Rudin 2 Index
15. **S_dbw**: <i> M. Halkidi and M. Vazirgiannis, "Clustering validity assessment: finding the optimal partitioning of a 
data set," Proceedings 2001 IEEE International Conference on Data Mining. </i>
16. **sd_dis Index**: <i>Halkidi et al. On clustering validation techniques. J. Intell. Inf. Syst., 2001.</i>
17. **sd_scat Index**: <i>Halkidi et al. On clustering validation techniques. J. Intell. Inf. Syst., 2001.</i> 

18. **pbm**: <i> Bandyopadhyay S. Pakhira M. K. and Maulik U. Validity index for crisp and fuzzy clusters. Pattern 
             Recognition, 2004. </i>
19. ratkowsky_lance
20. 
21. **ray_turi**: <i> Ray et al. Determination of number of clusters in k-means clustering and application in colour 
                  image segmentation. 4th International Conference on Advances in Pattern Recognition and Digital 
                  Techniques, 1999. </i>
22. wemmert_gancarski
23. **xie_beni**: <i> X.L. Xie and G. Beni. A validity measure for fuzzy clustering. IEEE Transactions on Pattern 
                  Analysis and Machine Intelligence, 1991. </i>
24. 
25. banfeld_raftery
26. trace_wib
27. 
28. log_det_ratio
29. 
30. point_biserial
31. calinski_harabasz
32. silhouette
33. davies_bouldin
34. scott_symons
