# -*- coding: utf-8 -*-

# Load data

import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB

from google.colab import files
uploaded = files.upload()

"""Analysis of line drawings for PPA"""

PPA = pd.read_csv('S01_PPA.csv',sep=r',',skipinitialspace = True,index_col='type');
PPA

from numpy import character
# Split data into training and test

def split_data_og(df_name):
  df = df_name.loc['original']
  leftOutRun = df['run'][0]
  naturalCategories = ['beaches','forests','mountains']
  manmadeCategories = ['city','highways','offices']
  testData = df.loc[df['run'] == leftOutRun]
  testLabels = testData['category']
  binaryTestLabels = testLabels.replace(to_replace = naturalCategories,value = 'natural')\
                              .replace(to_replace = manmadeCategories,value = 'manmade').to_numpy()
  testLabels = testLabels.to_numpy()
  testSamples = testData.iloc[:,2:].to_numpy()
  trainData = df.loc[df['run'] != leftOutRun]
  trainLabels = trainData['category']
  binaryTrainLabels = trainLabels.replace(to_replace = naturalCategories,value = 'natural')\
                                .replace(to_replace = manmadeCategories,value = 'manmade').to_numpy()
  trainLabels = trainLabels.to_numpy()
  trainSamples = trainData.iloc[:,2:].to_numpy()
  return testSamples, binaryTestLabels, binaryTrainLabels, testLabels, trainLabels, trainSamples

def split_data_line(df_name):
  df = df_name.loc['lineDrawings']
  leftOutRun = df['run'][0]
  naturalCategories = ['beaches','forests','mountains']
  manmadeCategories = ['city','highways','offices']
  testData = df.loc[df['run'] == leftOutRun]
  testLabels = testData['category']
  binaryTestLabels = testLabels.replace(to_replace = naturalCategories,value = 'natural')\
                              .replace(to_replace = manmadeCategories,value = 'manmade').to_numpy()
  testLabels = testLabels.to_numpy()
  testSamples = testData.iloc[:,2:].to_numpy()
  trainData = df.loc[df['run'] != leftOutRun]
  trainLabels = trainData['category']
  binaryTrainLabels = trainLabels.replace(to_replace = naturalCategories,value = 'natural')\
                                .replace(to_replace = manmadeCategories,value = 'manmade').to_numpy()
  trainLabels = trainLabels.to_numpy()
  trainSamples = trainData.iloc[:,2:].to_numpy()
  return testSamples, binaryTestLabels, binaryTrainLabels, testLabels, trainLabels, trainSamples

testSamples, binaryTestLabels, binaryTrainLabels, testLabels, trainLabels, trainSamples = split_data_line(PPA)

# Train the SVM Classifier

from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(trainSamples, binaryTrainLabels)
binaryPredictLabels = clf.predict(testSamples)

# Check the prediction

print('Predicted binary labels:')
print(binaryPredictLabels)
print('True binary labels:')
print(binaryTestLabels)

# Classification report

import sklearn.metrics as sm
print(sm.classification_report(binaryTestLabels,binaryPredictLabels))

# make binary labels for precision-recall and ROC curves
y_true = []; y_pred = []
for i in range(testLabels.size):
  y_true.append(binaryTestLabels[i] == 'manmade')
  y_pred.append(binaryPredictLabels[i] == 'manmade')

sm.PrecisionRecallDisplay.from_predictions(y_true,y_pred)
sm.RocCurveDisplay.from_predictions(y_true,y_pred)

"""Repeated analysis for photographs and line drawings for V1, V2,
V4, and RSC
"""

V1 = pd.read_csv('S01_V1.csv',sep=r',',skipinitialspace = True,index_col='type')
V2 = pd.read_csv('S01_V2.csv',sep=r',',skipinitialspace = True,index_col='type')
V4 = pd.read_csv('S01_V4.csv',sep=r',',skipinitialspace = True,index_col='type')
RSC = pd.read_csv('S01_RSC.csv',sep=r',',skipinitialspace = True,index_col='type')

"""Added in other classifiers as well (multiclass ovo, multiclass ovr, naive bayes)

Original photogaphs:
"""

# Commented out IPython magic to ensure Python compatibility.
ROIs = [PPA, V1, V2, V4, RSC]

for df in ROIs:
  testSamples, binaryTestLabels, binaryTrainLabels, testLabels, trainLabels, trainSamples = split_data_og(df)
  print(sm.classification_report(binaryTestLabels,binaryPredictLabels))
  # make binary labels for precision-recall and ROC curves
  y_true = []; y_pred = []
  for i in range(testLabels.size):
    y_true.append(binaryTestLabels[i] == 'manmade')
    y_pred.append(binaryPredictLabels[i] == 'manmade')
  sm.PrecisionRecallDisplay.from_predictions(y_true,y_pred)
  sm.RocCurveDisplay.from_predictions(y_true,y_pred)

  # Try multiclass, ovo
  clf.fit(trainSamples,trainLabels)  # this is doing one-versus-one
  multiPredictLabels = clf.predict(testSamples)
  print('Predicted labels:')
  print(multiPredictLabels)
  print('True labels:')
  print(testLabels)
  print(sm.classification_report(testLabels,multiPredictLabels))
  sm.ConfusionMatrixDisplay.from_predictions(testLabels,multiPredictLabels)

  # Try multiclass, ovr
  lin_clf = svm.LinearSVC()
  lin_clf.fit(trainSamples,trainLabels)
  linPredictLabels = lin_clf.predict(testSamples)

  print(sm.classification_report(testLabels,linPredictLabels))
  sm.ConfusionMatrixDisplay.from_predictions(testLabels,linPredictLabels)

  # Try Naive Bayes
  gnb = GaussianNB()
  y_pred = gnb.fit(trainSamples, binaryTrainLabels).predict(testSamples)
  print("Number of mislabeled points out of a total %d points : %d"
#     % (testSamples.shape[0], (binaryTestLabels != y_pred).sum()))

"""Line drawings:"""

# Commented out IPython magic to ensure Python compatibility.
for df in ROIs:
  testSamples, binaryTestLabels, binaryTrainLabels, testLabels, trainLabels, trainSamples = split_data_line(df)
  print(sm.classification_report(binaryTestLabels,binaryPredictLabels))
  # make binary labels for precision-recall and ROC curves
  y_true = []; y_pred = []
  for i in range(testLabels.size):
    y_true.append(binaryTestLabels[i] == 'manmade')
    y_pred.append(binaryPredictLabels[i] == 'manmade')
  sm.PrecisionRecallDisplay.from_predictions(y_true,y_pred)
  sm.RocCurveDisplay.from_predictions(y_true,y_pred)

  # Try multiclass, ovo
  clf.fit(trainSamples,trainLabels)  # this is doing one-versus-one
  multiPredictLabels = clf.predict(testSamples)
  print('Predicted labels:')
  print(multiPredictLabels)
  print('True labels:')
  print(testLabels)
  print(sm.classification_report(testLabels,multiPredictLabels))
  sm.ConfusionMatrixDisplay.from_predictions(testLabels,multiPredictLabels)

  # Try multiclass, ovr
  lin_clf = svm.LinearSVC()
  lin_clf.fit(trainSamples,trainLabels)
  linPredictLabels = lin_clf.predict(testSamples)

  print(sm.classification_report(testLabels,linPredictLabels))
  sm.ConfusionMatrixDisplay.from_predictions(testLabels,linPredictLabels)

  # Try Naive Bayes
  gnb = GaussianNB()
  y_pred = gnb.fit(trainSamples, binaryTrainLabels).predict(testSamples)
  print("Number of mislabeled points out of a total %d points : %d"
#     % (testSamples.shape[0], (binaryTestLabels != y_pred).sum()))

"""# **Trying different classifiers**

The worst classifiers for this dataset were the svm multiclass 1-v-1 and 1-v-r classifiers, with an average accuracy of ~20%. The best classifiers were the Naive Bayes and linear SVM classifiers (50% accuracy). The differences between the two optimal classifiers is that the NB algorithm is very simple (treats features independently) whereas the SVM algorithm is more complex (looks at interactions). Moreover, one is probabilistic while the other is geometric. If the number of images in the training and testing sets increase, the difference in accuracy between the two may be larger (with SVM having higher accuracy). However, with small datasets past work has shown that NB may outperform SVM.
"""
