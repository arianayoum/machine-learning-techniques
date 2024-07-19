# -*- coding: utf-8 -*-
"""
Load the data
"""

import os
import pandas as pd
import numpy as np
from google.colab import files
from sklearn.model_selection import permutation_test_score
uploaded = files.upload()

# import relevant packages
from sklearn.pipeline import make_pipeline
import sklearn.metrics as sm
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from scipy import stats

"""# Permutation analysis - PPA

Load data
"""

from sklearn.model_selection import permutation_test_score
import random
numSubj = 10
all_perm_scores = []
all_class_scores = []
all_pvals = []

for subj in range(1,numSubj+1):
  # construct file name
  filename = 'S%02d_PPA.csv' % subj
  print('Processing Subj %d out of %d: %s' % (subj,numSubj,filename))

  # load data
  allData = pd.read_csv(filename,sep=r',',skipinitialspace = True,index_col='type');

  # only use line data
  lineData = allData.loc['lineDrawings']
  labels = lineData['category'].to_numpy()
  samples = lineData.iloc[:,2:].to_numpy()
  runIdx = lineData['run'].to_numpy()

  # set up cross validation classification
  numSplits = len(np.unique(runIdx))
  CVfolds = GroupKFold(n_splits=numSplits)
  clf = make_pipeline(StandardScaler(),svm.SVC(kernel='linear'))

  # run permutation analysis
  random.seed(1000)
  score,permutation_scores,pvalue = permutation_test_score(clf,samples,labels,groups=runIdx,scoring='accuracy',cv = CVfolds, n_permutations=1000,n_jobs = 4)

  # group-average classification score
  all_class_scores.append(score)
  avg_class_score = np.average(all_class_scores)

  # group-average p value
  all_pvals.append(pvalue)
  avg_pvals = np.average(all_pvals)

  # compute group-average accuracy for each iteration of the permutation
  all_perm_scores.append(permutation_scores)

avg_perm_scores = np.array(np.average(all_perm_scores, axis=0))

"""Run Permutation analysis

Plot distribution
"""

import matplotlib.pyplot as plt
chance = 1/6
plt.hist(avg_perm_scores, 20, label='Permutation scores averaged across subjects')
ylim = plt.ylim()
plt.vlines(avg_class_score, ylim[0], ylim[1], linestyle='--',
          color='g', linewidth=3, label='Classification Score'
          ' (pvalue %s)' % avg_pvals)
plt.vlines(chance, ylim[0], ylim[1], linestyle='--',
          color='k', linewidth=3, label='Chance')

plt.ylim(ylim)
plt.legend()
plt.xlabel('Score')
plt.show()

"""# *Evaluate the distribution of the average scores: What fraction of permutations is greater than the group-mean of the correct ordering of the labels?*

## The p-value is 0.02; in other words, if the classifier is not better than chance, then obtaining a difference in performance accuracy occurs with a probability of only 2%. That's a very low probability! It seems enough to reject our null hypothesis and accept the alternative: the classifier performs with an accuracy that is above chance.

# Permutation Analysis - V1
"""

from sklearn.model_selection import permutation_test_score
import random
numSubj = 10
V1_all_perm_scores = []
V1_all_class_scores = []
V1_all_pvals = []

for subj in range(1,numSubj+1):
  # construct file name
  filename = 'S%02d_V1.csv' % subj
  print('Processing Subj %d out of %d: %s' % (subj,numSubj,filename))

  # load data
  V1_allData = pd.read_csv(filename,sep=r',',skipinitialspace = True,index_col='type');

  # only use line data
  lineData = V1_allData.loc['lineDrawings']
  labels = lineData['category'].to_numpy()
  samples = lineData.iloc[:,2:].to_numpy()
  runIdx = lineData['run'].to_numpy()

  # set up cross validation classification
  numSplits = len(np.unique(runIdx))
  CVfolds = GroupKFold(n_splits=numSplits)
  clf = make_pipeline(StandardScaler(),svm.SVC(kernel='linear'))

  # run permutation analysis
  random.seed(1000)
  score,permutation_scores,pvalue = permutation_test_score(clf,samples,labels,groups=runIdx,scoring='accuracy',cv = CVfolds, n_permutations=1000,n_jobs = 4)

  # group-average classification score
  V1_all_class_scores.append(score)
  V1_avg_class_score = np.average(V1_all_class_scores)

  # group-average p value
  V1_all_pvals.append(pvalue)
  V1_avg_pvals = np.average(V1_all_pvals)

  # compute group-average accuracy for each iteration of the permutation
  V1_all_perm_scores.append(permutation_scores)

V1_avg_perm_scores = np.array(np.average(V1_all_perm_scores, axis=0))

import matplotlib.pyplot as plt

plt.hist(V1_avg_perm_scores, 20, label='Permutation scores averaged across subjects')
ylim = plt.ylim()
plt.vlines(V1_avg_class_score, ylim[0], ylim[1], linestyle='--',
          color='g', linewidth=3, label='Classification Score'
          ' (pvalue %s)' % V1_avg_pvals)
plt.vlines(chance, ylim[0], ylim[1], linestyle='--',
          color='k', linewidth=3, label='Chance')

plt.ylim(ylim)
plt.legend()
plt.xlabel('Score')
plt.show()

"""# *Evaluate the distribution of the average scores: What fraction of permutations is greater than the group-mean of the correct ordering of the labels?*

## The p-value is 0.04; in other words, if the classifier is not better than chance, then obtaining a difference in performance accuracy occurs with a probability of only 4%. That's a very low probability! It seems enough to reject our null hypothesis and accept the alternative: the classifier performs with an accuracy that is above chance.

# In conclusion, both the classifiers for V1 and the PPA perform significantly above chance. These results took ~45 minutes to run. The larger the number of iterations, the longer it takes to run. Permutation tests are very useful especially in this case (as the sample size was so small; n=10) since it shuffles among existing values and doesn't make any assumptions of normality.
"""
