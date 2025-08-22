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

1. **ball_hall Index**: <i> G. H. Ball and D. J. Hall. Isodata: A novel method of data analysis and pattern
                      classification. Menlo Park: Stanford Research Institute. (NTIS No. AD 699616),1965.</i>
2. **banfeld_raftery Index**: <i> J.D. Banfield and A.E. Raftery. Model-based gaussian and non-gaussian clustering. Biometrics,
                        49:803–821, 1993. </i>

3. **c_index Index**: <i> Hubert, Lawrence & Levin, Joel. (1976). A general statistical framework for assessing categorical 
clustering in free recall. Psychological Bulletin. 83. 1072-1080. 10.1037/0033-2909.83.6.1072. </i>
4. **Calinski-Harabasz Index**: <i>T. Calinski and J. Harabasz. A dendrite method for cluster analysis. Communications in 
                             Statistics, 3, no. 1:1–27, 1974. </i>
5. **CDbw Index** : <i>Halkidi, M., & Vazirgiannis, M. (2008). A density-based cluster validity approach using 
multi-representatives. Pattern Recognit. Lett., 29, 773-786.  </i>

6. **Davies-Bouldin Index**: <i> D. L. Davies and D. W. Bouldin. A cluster separation measure. IEEE Transactions on 
                           Pattern Analysis and Machine Intelligence, PAMI-1, no. 2:224–227, 1979.</i>
7. **det_ratio Index** : <i> A. J. Scott and M. J. Symons. Clustering methods based on likelihood ratio criteria. Biometrics, 
                27:387–397, 1971.</i>
8. **Dunn Index** : <i>J. Dunn. Well separated clusters and optimal fuzzy partitions. Journal of Cybernetics, 4:95–104, 
                    1974. </i>

9. **g+ Index**: <i> F. J. Rohlf. Methods of comparing classifications. Annual Review of Ecology and Systematics, 5:101–113, 
               1974.</i>
10. **gamma Index**: <i> F. B. Baker and L. J. Hubert. Measuring the power of hierarchical cluster analysis. Journal of the 
                   American Statistical Association, 70:31–38, 1975. </i>
11. **GDI (11,...,61)(12,...,62)(13,...,63) Indexes**: <i>J. C. Bezdek and N. R. Pal. Some new indexes 
of cluster validity. IEEE Transactions on Systems, Man, and CyberneticsÑPART B: CYBERNETICS, 28, no.3:301–315, 1998.</i>

12. **ksq_detw Index**:  F. H. B. Marriot. Practical problems in a method of cluster analysis. Biometrics,
27:456–460, 1975.

13. **log_det_ratio Index**: <i> Halkidi et al. On clustering validation techniques. J. Intell. Inf. Syst., 2001. </i>
14. **log_ss_ratio Index**: <i> J. A. Hartigan. Clustering algorithms. New York: Wiley, 1975. </i>

15. **McClain_Rao Index**: <i> J. O. McClain and V. R. Rao. Clustisz: A program to test for the quality of
                         clustering of a set of objects. Journal of Marketing Research, 12:456–460, 1975.</i>

16. **pbm Index**: <i> Bandyopadhyay S. Pakhira M. K. and Maulik U. Validity index for crisp and fuzzy clusters. Pattern 
             Recognition, 2004. </i>
17. **point_biserial Index**: <i>G. W. Milligan. A monte carlo study of thirty internal criterion measures for
                           cluster analysis. Psychometrika, 46, no. 2:187–199, 1981. </i>

18. **ray_turi Index**: <i> Ray et al. Determination of number of clusters in k-means clustering and application in colour 
                  image segmentation. 4th International Conference on Advances in Pattern Recognition and Digital 
                  Techniques, 1999. </i>
19. **ratkowsky_lance Index**: <i> D. A. Ratkowsky and G. N. Lance. A criterion for determining the number of
                             groups in a classification. Australian Computer Journal, 10:115–117, 1978.</i>

20. **scot_symmons index**: <i> X.L. Xie and G. Beni. A validity measure for fuzzy clustering. IEEE Transactions
                          on Pattern Analysis and Machine Intelligence, 13(4):841–846, 1991. </i>
21. **sd_scat Index**: <i>Halkidi et al. On clustering validation techniques. J. Intell. Inf. Syst., 2001.</i> 
22. **sd_dis Index**: <i>Halkidi et al. On clustering validation techniques. J. Intell. Inf. Syst., 2001.</i>
23. **S_dbw Index**: <i> M. Halkidi and M. Vazirgiannis, "Clustering validity assessment: finding the optimal partitioning of a 
data set", Proceedings 2001 IEEE International Conference on Data Mining. </i>
24. **silhouette index**: <i>Rousseeuw P.J. Silhouettes: a graphical aid to the interpretation and validation of
                       cluster analysis. Journal of Computational and Applied Mathematics, 20:53–65,
                       1987 </i>

25. **trace_w Index**: <i> A. W. F. Edwards and L. Cavalli-Sforza. A method for cluster analysis.
                           Biometrika, 56:362–375, 1965. </i>
26. **trace_wib Index**: <i> H. P. Friedman and J. Rubin. On some invariant criteria for grouping data. Journal
                       of the American Statistical Association, 62:1159–1178, 1967.</i>
27. **tau Index**: <i> Zhu, Erzhou, Xue Wang, and Feng Liu. "A new cluster validity index for overlapping datasets." 
                 Journal of Physics: Conference Series. Vol. 1168. No. 3. IOP Publishing, 2019.</i> 

28. **wemmert_gancarski Index**: <i> J. Zhang, T. Nguyen, S. Cogill, A. Bhatti, L. Lingkun, S. Yang, S. Nahavandi. (2018). A review on cluster estimation methods and their application to neural spike data. Journal of Neural Engineering.</i> 
29. **xie_beni Index**: <i> X.L. Xie and G. Beni. A validity measure for fuzzy clustering. IEEE Transactions on Pattern 
                  Analysis and Machine Intelligence, 1991. </i>

