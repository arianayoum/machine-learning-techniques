# -*- coding: utf-8 -*-
"""Neural-network-classifier.ipynb

Load data
"""

from sklearn.datasets import fetch_20newsgroups
groups = fetch_20newsgroups() # this can take a couple of seconds

groups.keys() # structure of the data

groups.target # target keys correspond to a newsgroup, encoded as an integer

# What are the distinct values of the targets?
import numpy as np
np.unique(groups.target)

import seaborn as sb
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction import text
nltk.download('names')
name_words = nltk.corpus.names.words()
print(name_words[0:20]);
all_names = set(name_words)

nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# initialize a cleaned doc list
data_cleaned = []

# loop over newsgroups contributions
for doc in groups.data:
  doc = doc.lower() # convert to lower case only
  doc_cleaned = ' '.join(lemmatizer.lemmatize(word)
                        for word in doc.split()
                        if word.isalpha() and
                        word not in all_names)
  data_cleaned.append(doc_cleaned)

from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer(stop_words="english", max_features=500)

# now count up the cleaned data
data_cleaned_count = count_vector.fit_transform(data_cleaned)

categories_4 = ['alt.atheism','talk.religion.misc','comp.graphics','sci.space']
groups_4 = fetch_20newsgroups(subset='all',categories=categories_4)

def is_letter_only(word):
  for letter in word:
    if not letter.isalpha():
      return False
  return True

data_cleaned_4 = []
for doc in groups_4.data:
  doc = doc.lower()
  doc_cleaned = ' '.join(lemmatizer.lemmatize(word)
                        for word in doc.split()
                        if is_letter_only(word) and
                        word not in all_names)
  data_cleaned_4.append(doc_cleaned)

# minumum document frequency to be considered: 2; maximum: 50% of the dataset
count_vector = CountVectorizer(stop_words="english", max_features=None, max_df=0.5, min_df=2)

data = count_vector.fit_transform(data_cleaned_4)

"""# 1. Perform k-means clustering on the 4-groups Newsgroup data using different values of k.

"""

from sklearn.cluster import KMeans
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vector = TfidfVectorizer(stop_words="english",max_features=None, max_df=0.5, min_df=2)
data = tfidf_vector.fit_transform(data_cleaned_4)

from collections import Counter
kmeans.fit(data)
clusters = kmeans.labels_
print(Counter(clusters))

Sum_of_squared_distances = []
Sum_of_squared_distances.append(kmeans.inertia_)
kmeans.inertia_

from sklearn.cluster import KMeans
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)

kmeans.fit(data)
clusters = kmeans.labels_
print(Counter(clusters))

Sum_of_squared_distances.append(kmeans.inertia_)
kmeans.inertia_

from sklearn.cluster import KMeans
k = 2
kmeans = KMeans(n_clusters=k, random_state=42)

kmeans.fit(data)
clusters = kmeans.labels_
print(Counter(clusters))

Sum_of_squared_distances.append(kmeans.inertia_)
kmeans.inertia_

from sklearn.cluster import KMeans
k = 1
kmeans = KMeans(n_clusters=k, random_state=42)

kmeans.fit(data)
clusters = kmeans.labels_
print(Counter(clusters))

Sum_of_squared_distances.append(kmeans.inertia_)
kmeans.inertia_

"""# 2. Plot the sum of squared errors (SSE) as a function of k. What do you observe?"""

import matplotlib.pyplot as plt

k=[4,3,2,1]

# Visualization
plt.plot(k, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('SSE')
plt.show()

"""# This plot indicates that as k (or the number of clusters) increase, the SSE decreases.

# 3. Try different numbers of topics for the LDA analysis. Which one produces more meaningful topics in the end?
"""

from sklearn.decomposition import LatentDirichletAllocation
t = 20 # number of topics
lda = LatentDirichletAllocation(n_components=t, learning_method='batch', random_state=42)
data_cleaned_count = count_vector.fit_transform(data_cleaned_4)
lda.fit(data_cleaned_count)

terms = count_vector.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
  print("Topic {}:".format(topic_idx))
  print(" ".join([terms[i] for i in topic.argsort()[-10:]]))

from sklearn.decomposition import LatentDirichletAllocation
t = 5 # number of topics
lda = LatentDirichletAllocation(n_components=t, learning_method='batch', random_state=42)
data_cleaned_count = count_vector.fit_transform(data_cleaned_4)
lda.fit(data_cleaned_count)

terms = count_vector.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
  print("Topic {}:".format(topic_idx))
  print(" ".join([terms[i] for i in topic.argsort()[-10:]]))

from sklearn.decomposition import LatentDirichletAllocation
t = 10 # number of topics
lda = LatentDirichletAllocation(n_components=t, learning_method='batch', random_state=42)
data_cleaned_count = count_vector.fit_transform(data_cleaned_4)
lda.fit(data_cleaned_count)

terms = count_vector.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
  print("Topic {}:".format(topic_idx))
  print(" ".join([terms[i] for i in topic.argsort()[-10:]]))

from sklearn.decomposition import LatentDirichletAllocation
t = 30 # number of topics
lda = LatentDirichletAllocation(n_components=t, learning_method='batch', random_state=42)
data_cleaned_count = count_vector.fit_transform(data_cleaned_4)
lda.fit(data_cleaned_count)

terms = count_vector.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
  print("Topic {}:".format(topic_idx))
  print(" ".join([terms[i] for i in topic.argsort()[-10:]]))

from sklearn.decomposition import LatentDirichletAllocation
t = 40 # number of topics
lda = LatentDirichletAllocation(n_components=t, learning_method='batch', random_state=42)
data_cleaned_count = count_vector.fit_transform(data_cleaned_4)
lda.fit(data_cleaned_count)

terms = count_vector.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
  print("Topic {}:".format(topic_idx))
  print(" ".join([terms[i] for i in topic.argsort()[-10:]]))

"""# It seems like as the topics increase in the LDA analysis, each topic group becomes more specific and thus meaningful. For example, when there's only 10 groupings, the word "article" is included in the Christianity topic, but in 30 groupings, there's mostly only very relevant words such as "belief say people bible atheist believe christian jesus"

# 4. Experiment with processing the entire 20 groups of the Newsgroup data with LDA. Are the resulting topics full of noise or gems? Please comment.
"""

from sklearn.decomposition import LatentDirichletAllocation
t = 20 # number of topics
lda = LatentDirichletAllocation(n_components=t, learning_method='batch', random_state=42)
data_cleaned_count = count_vector.fit_transform(data_cleaned)
lda.fit(data_cleaned_count)

terms = count_vector.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
  print("Topic {}:".format(topic_idx))
  print(" ".join([terms[i] for i in topic.argsort()[-10:]]))

"""# I ran this first exploratory LDA with just 20 groupings and it's pretty noisy although not terrible. There seems to be a relevant topic for each grouping, for example in the first grouping the words "institute, university, problem, article" all seem to go together but it's unclear how certain words such as "sale" or "doe" fit in. There's probably not enough groupings. In the next part, I try re-running it with 100 groupings."""

from sklearn.decomposition import LatentDirichletAllocation
t = 100 # number of topics
lda = LatentDirichletAllocation(n_components=t, learning_method='batch', random_state=42)
data_cleaned_count = count_vector.fit_transform(data_cleaned)
lda.fit(data_cleaned_count)

terms = count_vector.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
  print("Topic {}:".format(topic_idx))
  print(" ".join([terms[i] for i in topic.argsort()[-10:]]))

"""# The data looks even noisier than before. It may be that it doesn't necessarily get better with a higher grouping number. There may be a sweet spot where there's enough topics that are relevant to the pool of words but aren't too many so that the pool of words have to be split up randomly."""
