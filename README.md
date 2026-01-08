Code for the Journal of Cognitive Neuroscience article "Representing Context and Priority in Working Memory" (https://direct.mit.edu/jocn/article-abstract/36/7/1374/120857/Representing-Context-and-Priority-in-Working?redirectedFrom=fulltext).

‘FiringRateRecurrentNeuralNetwork_DSR.ipynb’: A Google Colab notebook training and testing the RNNs, and plots dynamic visualizations of the networks via PCA.

‘TVI_context_ytp.ipynb’‘TVI_priority_ytp.ipynb’: Perform context and priority-based TVI analyses on the fMRI data from Yu, Teng, & Postle, 2020.

‘TVI_context_DSREEG.ipynb’‘TVI_priority_DSREEG.ipynb’: Perform context and priority-based TVI analyses on the EEG data from Fulvio & Postle, 2020.

‘svm_fMRI.m’: Matlab script performing within- and cross-label decoding anlaysis of the fMRI data. Also produces decoding accuracy plots.

‘svm_RNN_packaged’: Performs within- and cross-label decoding of the RNN recurrent unit activities.

‘fdr_bh.m’: Helper script to perform FDR correction for multiple comparisons.

‘plot_areaerrorbar’: Helper script to plot error bars in figures.
