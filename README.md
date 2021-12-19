# CSCI5523_Final_Project
This is a repository for code and files used for our final project


All scripts & files found in the SVM_Scripts and SVM_Data_Files folders were gathered from / adapted from the papers:

Golub TR, Slonim DK, Tamayo P, Huard C, Gaasenbeek M, Mesirov JP, Coller H, Loh ML, Downing JR, Caligiuri MA, Bloomfield CD, Lander ES. Molecular classification of cancer: class discovery and class prediction by gene expression monitoring. Science. 1999 Oct 15;286(5439):531-7. doi: 10.1126/science.286.5439.531. PMID: 10521349.

Li Z, Xie W, Liu T (2018) Efficient feature selection and classification for microarray data. PLoS ONE 13(8): e0202167. https://doi.org/10.1371/journal.pone.0202167




Necessary files for running the scripts in SVM_Scripts can be found in SVM_Data_Files.  Here is a description of the functions of the scripts:

  **svm_data_preprocessing.py** - used to preprocess and balance the data.
  
  **feature_selector.py** - used to select which features are used for classification.
  
  **mRMR.py** - alternative feature selection method.
  
  **Random_Forest.py** - comparison method of classification.
  
  **linear_svm.py** - linear kernal for svm model **warning** takes a very long time to run.
  
  **svm_rfe.py** - recursive feature elimination svm - used to remove unnecessary features.
  
  **svm_vssrfe.py** - variable step size recursive feature elimination - a more efficient method for feature elimination.
  
  **comparison_plots.py** - plot to compare the performance between feature selectors and classifiers.
