# The PyClustKit Module: All about clustering in a single Python Module!

The pyclustkit module is built to include various state-of-the-art meta-features for AutoML operatios in clustering, 
specifically meta-learning. Additionally the library includes the implementation of many clustering validity indices 
(CVIs) optimized for calculating many of them. 

While the python module of pyclustkit is the main option to use it programmatically, it can also be used 
through through the dedicated gradio-based user-interface also provided as a docker image. Instructions are presented 
in the relevant section. 

## Table of Contents
- [Python Module](#python-module)
  - [Installation Instructions](#installation-instructions)
- Framework (GUI)
- Implemented MF and CVI
- [Cite Us!](#citing-this-work)

- [Useful Links](#useful-links)
- [Usage Examples](#usage-examples)
- [GUI Version](#gui-version)
- 

# Python Module
The default way to use pyclustkit is as a library downloaded either as a pypi or by cloning this github repository. 

## Installation Instructions
To download from the pypi channel we can use the following command and start right away.

```commandline
pip install pyclustkit
```

Alternatively you can clone this repository to run the module. 

```commandline
git clone https://github.com/yannispoulakis/pyclustkit.git
cd pyclustkit
pip install -r requirements.txt
```

 ⚠️ **Warning** : Required version of dgl==2.2.1 may not be found in the pypi index when using Ubuntu/Debian Releases. 
This is an issue to be fixed in future versions of the library. For now you can use 

```commandline
pip install pyclustkit -f https://data.dgl.ai/wheels/torch-2.3/repo.html
```


# Usage Examples
## Calculating Internal Cluster Validity Indices (CVI) 

PyClustKit comes with an evaluation suite of 45 internal validity indices. Each is implemented on top of numpy and, 
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

12. **ksq_detw Index**: <i> F. H. B. Marriot. Practical problems in a method of cluster analysis. Biometrics,
27:456–460, 1975. </i>

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
                       1987. </i>

25. **trace_w Index**: <i> A. W. F. Edwards and L. Cavalli-Sforza. A method for cluster analysis.
                           Biometrika, 56:362–375, 1965. </i>
26. **trace_wib Index**: <i> H. P. Friedman and J. Rubin. On some invariant criteria for grouping data. Journal
                       of the American Statistical Association, 62:1159–1178, 1967.</i>
27. **tau Index**: <i> Zhu, Erzhou, Xue Wang, and Feng Liu. "A new cluster validity index for overlapping datasets." 
                 Journal of Physics: Conference Series. Vol. 1168. No. 3. IOP Publishing, 2019.</i> 

28. **wemmert_gancarski Index**: <i> J. Zhang, T. Nguyen, S. Cogill, A. Bhatti, L. Lingkun, S. Yang, S. Nahavandi. A review on cluster estimation methods and their application to neural spike data. Journal of Neural Engineering, 2018.</i> 
29. **xie_beni Index**: <i> X.L. Xie and G. Beni. A validity measure for fuzzy clustering. IEEE Transactions on Pattern 
                  Analysis and Machine Intelligence, 1991. </i>
</details>

### Implemented Meta-Features
<details>
<summary>List of Implemented Meta-Features with citations</summary>
Currently the collection consists of the following Meta-Features. We group the Meta-Features based on the paper that first introduced them:

<details>
<summary>
M. C. P. de Souto, R. B. C. Prudêncio, R. F. Soares, D. S.A. de Araujo, I. G.Costa, T. B. Ludermir, and A. Schliep. Ranking and selecting clustering algorithms using a meta-learning approach. In Proceedings of the International Joint Conference on Neural Networks, IJCNN 2008, part of the IEEE World Congress on Computational Intelligence, WCCI 2008, Hong Kong, China, June 1-6, 2008, pages 3729–3735. IEEE, 2008.
</summary>
<ol>
  <li>(Log)No.instances.</li>
  <li>(Log) Ratio of instances to the number of features.</li>
  <li>% of missing values.</li>
  <li>Proportion of 𝑇^2 that are within 50% of the Chi-Squared distribution.</li>
  <li>Skewness of the 𝑇^2 vector.</li>
  <li>% of attributes kept after selection filter.</li>
  <li> % of outliers *.</li>
</ol>
</details>

<details>
<summary>
A. C. A. Nascimento, R. B. C. Prudêncio, M. C. P. de Souto, and I. G. Costa. Mining rules for the automatic selection process of clustering methods applied to cancer gene expression data. In Cesare Alippi, Marios M. Polycarpou, Christos G. Panayiotou, and Georgios Ellinas, editors, Artificial Neural Networks - ICANN 2009, 19th International Conference, Limassol, Cyprus, September 14-17, 2009, Proceedings, Part II, volume 5769 of Lecture Notes in Computer Science, pages 20–29. Springer, 2009.</summary>
<ol>
  <li>No. clusters with size inferior to k.</li>
  <li>No. clusters with size superior to k.</li>
  <li>Normalized Relative Entropy of cluster distribution.</li>
  <li>Classification error obtained by the KNN algorithm (k3).</li>
</ol>
</details>

<details>
<summary>D. G. Ferrari and L. N. de Castro. Clustering algorithm selection by meta-learning systems: A new distance-based problem characterization and ranking combination methods. Inf. Sci., 301:181–194, 2015.</summary>
<ol>
  <li>(Log) No. features.</li>
  <li>% of discrete attributes.</li>
  <li>(Min, Max, Mean, std) class skewness.</li>
  <li>(Min, Max, Mean, std) class kurtosis.</li>
  <li>(Min, Max, Mean, std) discrete feature entropy.</li>
  <li>Mean feature correlation.</li>
  <li>Mean discrete feature concentration.</li>
  <li>Mean, Variance, Std. Deviation, Skewness, Kurtosis of the distance vector d.</li>
  <li>% of values in 10 formed bins of equal size in the range of the normalized distance vector d.</li>
  <li>% of values with absolute zscore of the normalized distance vector in 4 formed bins.</li>
</ol>
</details>

<details>
<summary>M. Vukicevic, S. Radovanovic, B. Delibasic, and M. Suknovic. Extending meta-learning framework for clustering gene expression data with component-based algorithm design and internal evaluation measures. Int. J. Data Min. Bioinform., 14(2):101–119, 2016.</summary>
<ol>
  <li>Internal CVIs as measured by each algorithm configuration.</li>
  <li>Reusable Components, 1 meta-feature for each of the table x.</li>
</ol>
</details>

<details>
<summary>B. A. Pimentel and A. C. P. L. F. de Carvalho. A new data characterization for selecting clustering algorithms using meta-learning. Inf. Sci., 477:203–219, 2019.</summary>
<ol>
  <li>Mean, Variance, Std. Deviation, Skewness, Kurtosis of the correlation vector c.</li>
  <li>% of values in 10 formed bins of equal size in the range of the normalized correlation vector c.</li>
  <li>% of values with absolute z score of the normalized correlation vector c in 4 formed bins</li>
</ol>
</details>

<details>
<summary>Y. Poulakis, C. Doulkeridis, and D. Kyriazis. Autoclust: A framework for automated clustering based on cluster validity indices. In Claudia Plant, Haixun Wang, Alfredo Cuzzocrea, Carlo Zaniolo, and Xindong Wu, editors, 20th IEEE International Conference on Data Mining, ICDM 2020, Sorrento, Italy, November 17-20, 2020, pages 1220–1225. IEEE, 2020.</summary>
<ol>
  <li>Internal CVIs measured by MeanShift</li>
</ol>
</details>

<details>
<summary>N. Cohen-Shapira and L. Rokach. Automatic selection of clustering algorithms using supervised graph embedding. Inf. Sci., 577:824–851, 2021.</summary>
<ol>
  <li>300 meta-features from the previous to ✗ ✗ last graph CNN layer.</li>
</ol>
</details>

<details>
<summary>Y. Liu, S. Li, and W. Tian. Autocluster: Meta-learning based ensemble method for automated unsupervised clustering. In Kamal Karlapalem, Hong Cheng, Naren Ramakrishnan, R. K. Agrawal, P. Krishna Reddy, Jaideep Srivastava, and Tanmoy Chakraborty, editors, Advances in Knowledge Discovery and Data Mining - 25th Pacific-Asia Conference, PAKDD 2021, Virtual Event, May 11-14, 2021, Proceedings, Part III, volume 12714 of Lecture Notes in Computer Science, pages 246–258. Springer, 2021.</summary>
<ol>
  <li>(Log) Ratio of features to the number of instances.</li>
  <li>Hopkins Statistic.</li>
  <li>PCA 95% deviations.</li>
  <li>Skewness & kurtosis of the 1st pc of PCA.</li>
  <li>Instance distance to closest cluster center (KMeans).</li>
  <li>No. leaves (Agglomerative Clustering).</li>
  <li>Reachability instances (OPTICS).</li>
  <li>Distances to become core points(OPTICS).</li>
</ol>
</details>

<details>
<summary>R. El Shawi and S. Sakr. Tpe-autoclust: A tree-based pipline ensemble framework for automated clustering. In K. Selçuk Candan, Thang N. Dinh, My T. Thai, and Takashi Washio, editors, IEEE International Conference on Data Mining Workshops, ICDM 2022 - Workshops, Orlando, FL, USA, November 28 - Dec. 1, 2022, pages 1144–1153. IEEE, 2022.</summary>
<ol>
  <li>Internal CVIs measured by MeanShift, DBSCAN, OPTICS.</li>
</ol>
</details>

</details>
