# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 09:40:55 2018

@author: Xu
"""
from __future__ import print_function
"""
This is a revised version of the previous patent analysis code

****20180218_PATENTTEXT_TFIDF_V1.00.py*****

by author: Salvatore Immordino 

with interactive visualizaiton added. The path to the files 
are given by getcwd(). Only some minor changes were made to the original code
only to make it work under Python 3.5. 

Following the original analysis code is the interactive visualization 
using Bokeh package being added in this version as of 04/15/2018. Instructions
on the interactive visualization is accompanied with that part of the code.

"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Title: IE596 – Text Mining and Analytics
File name: 20170404_TEXT.py
Author: Salvatore Immordino
Date created: Sun Nov 27 13:56:06 2016
Date last modified: 20170417
Python Version: 2.7

"""

from time import time
import numpy as np
import pandas as pd
import warnings
import string
# To make pretty charts
import matplotlib.pyplot as plt  # conda install -c anaconda seaborn=0.7.1
import matplotlib.cm as cm
import seaborn as sns
import pylab
import nltk
# To reduce dimensionionality
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
# To Clean and preprocess
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk import FreqDist, word_tokenize
from wordcloud import WordCloud
# To Vectorize & Transform
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# Package to cluster data
from sklearn.cluster import KMeans, MiniBatchKMeans
import sklearn.cluster as cluster
# To measure cluster performance
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import metrics
# from sklearn.metrics import pairwise_distances  # Calinski-Harabaz index

# ---------------------------------------------------------
#  Import data
# ---------------------------------------------------------
# Suppress deprecation warnings from pandas library
warnings.filterwarnings("ignore",
                        category=DeprecationWarning,
                        lineno=570)
# supress all warnings
warnings.filterwarnings("ignore")


#work = 'C:/Users/simmordino/'
#home = 'C:/Users/DAD/'
#home = 'C:/Users/Xu/'

# Set the file path
#path = 'Box Sync/Sam_Immordino_Thesis_Work/IE599_SPING_2017/20170409_DATA/'
#path = r'Desktop/Academics/data science/Patent analysis/'

import os
path=os.getcwd()

file_name = 'input_data2'  # Set the file_name

# Create raw pandas dataframe, set dtype to reduce memory demand
raw_data = pd.read_csv(path +
                       file_name +
                       '.csv',
                       encoding='iso-8859-1',
                       dtype={"patent_id": object,
                              "abstract": object,
                              "title": object},
                       header=0)
# Created a list of common patent terms, use soup to scavenge a glossary
patent_terms = pd.read_csv(path +
                           'patent_terms.csv',
                           encoding='iso-8859-1',
                           header=None)
# ---------------------------------------------------------
#  Preprocessing
# ---------------------------------------------------------
# create dataframe
df_raw = pd.DataFrame(raw_data, columns=['patent_id', 'abstract', 'title'])

# drop duplicates (1525 words)
df_drop_duplicates = df_raw.drop_duplicates()

# remove the nan's (1383 words)
df_dropna = df_drop_duplicates.dropna()

# combine abstract and title - maybe weight title higher?
df_dropna['combined'] = df_dropna[
        ['title', 'abstract']].apply(lambda x: ''.join(x), axis=1)

corpus = list(df_dropna.combined)
labels = list(df_dropna.patent_id)

nltk.download("stopwords")  # download set of stop words
nltk.download('punkt')      # download punctuation, check for hyphens?
nltk.download("wordnet")    # download for lemmatzation

stop = set(stopwords.words('english'))
punctuations = set(string.punctuation)
lemma = WordNetLemmatizer()
stemmer = SnowballStemmer("english", ignore_stopwords=True)
word_len = 4  # set length of words to remove
jargon = set(patent_terms.ix[:, 0])  # build a "patent jargon" lexical

# build and save patent terms list
'''need to use beutifull soup to scrape patent term glossary for bold words
   set max_df to get most frequuently used'''
jargon.add(u'comprising')
pd_jargon = pd.DataFrame(list(jargon))
pd_jargon.to_csv(home + path + 'patent_terms.csv', header=False, index=False)

# --------------------------------------------------------
#  Custome Functions
# ---------------------------------------------------------
'''https://stackoverflow.com/questions/13928155/spell-checker-for-python'''

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in punctuations)
    number_free = ''.join([c for c in punc_free if c not in "1234567890"])
    jargon_free = " " .join(
            [j for j in number_free.lower().split() if j not in jargon])
    lemmatized = " ".join(
            lemma.lemmatize(word) for word in jargon_free.split())
    smallword_free = ' '.join(
            [w for w in lemmatized.split() if len(w) > word_len])
    return smallword_free


def tokenizer(corpus):
    words = [i.strip(
            "".join(punctuations)) for i in word_tokenize(
                    str(corpus)) if i not in punctuations]
    return words


def stem_tokens(tokens, stemmer):  # snowball stemmer gave undesired results
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed  # tokens = stem_tokens(tokens, stemmer)  #


def byteify(input):  # remove the 'u's from unicode
    if isinstance(input, dict):
        return {byteify(key): byteify(value) for key,
                value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, str):
        return input.encode('utf-8')
    else:
        return input


def optimalK(data, nrefs=3, maxClusters=15):
    """
    Estimating # of clusters via the gap statistic..Tibshirani 2001..Zotero
    Calculates KMeans optimal K, Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    for gap_index, k in enumerate(range(1, maxClusters)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)
        # for n references, generate random sample and perform kmeans getting
        # resulting dispersion of each loop
        for i in range(nrefs):

            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)

        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append(
                {'clusterCount': k, 'gap': gap}, ignore_index=True)
        # Plus 1 cause index of 0 means 1 cluster is optimal,
        # index 2 = 3 are optimal
    return (gaps.argmax() + 1, resultsdf)

# ---------------------------------------------------------
#  Clean the data
# ---------------------------------------------------------
corpus_clean = [clean(doc).split() for doc in corpus]  # clean the corpus
corpus_uni = byteify(corpus_clean)  # remove unicode encoding for wordgram

# ---------------------------------------------------------
#  Tokenize words
# ---------------------------------------------------------
tokens = tokenizer(corpus_uni)

# ---------------------------------------------------------
#  Frequency analysis
# ---------------------------------------------------------
fd = FreqDist(tokens)
fd.most_common(40)
fd.plot(20)
fd.plot(20, cumulative=True)  # adds cummulative count from left to right

# generate a word cloud based on frequency
tokens_str = str(tokens)
tokens_str = tokens_str.replace("'", "")

# http://amueller.github.io/word_cloud/
wordcloud = WordCloud(max_font_size=40).generate(tokens_str)

plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")

# ---------------------------------------------------------
#  Intialization parameters
# ---------------------------------------------------------
# num_samples = 2000
num_features = 5000
sample_size = 1000
# num_topics = 10
num_clusters = 11
num_top_terms = 5  # top words closest for each cluster


# ---------------------------------------------------------
#  Vectorization
# ---------------------------------------------------------
# define count vectorizer parameters
count_vect = CountVectorizer(max_features=num_features,  # top words
                             max_df=0.65,  # ignore if used everywhere
                             min_df=0.0,  # ignore if only used once
                             tokenizer=lambda doc: doc,
                             lowercase=False,  # no lowercase; low error
                             ngram_range=(1, 3))  # unigram thru trigram

X_counts = count_vect.fit_transform(corpus_clean).todense()

# dictionary of feature indices
count_vect.vocabulary_.get(u'gypsum')

# features used in the dtm matrix. This is the word vocabulary
terms = count_vect.get_feature_names()  # check out possible start words
print(terms)  # first evidence of enabling words "Start words" e.g combination
len(terms)


# ---------------------------------------------------------
#  Term Frequency * Inverse Document Frequency, Tf-Idf
# ---------------------------------------------------------
'''
https://radimrehurek.com/gensim/tut2.html

expects a bag-of-words (integer values) training corpus during initialization.
1. Count words occurances by document and transform into a dtm or (X) - DONE
2. Apply Tf-idf weighting, words occur frequently in document not corpus - DONE

During transformation, it will take a vector and return another vector of the
same dimensionality, except that features which were rare in the training
corpus will have their value increased.  It therefore converts integer-valued
vectors into real-valued ones, while leaving the number of dimensions intact.

Term Frequency Inverse Document Frequency, or short tf-idf,
is a way to measure how important a term is in context of a document or corpus.
The importance increases proportionally to the number of times a word appears
in the document but is offset by the frequency of the word in the corpus.

'''
tfidf_transformer = TfidfTransformer()

# create document term matrix (dtm),
tf_idf_matrix = tfidf_transformer.fit_transform(X_counts)

# No sparse matrix
# X = tf_idf_matrix.todense() or tf_idf_matrix_dense = tf_idf_matrix.toarray()
tf_idf_matrix_dense = tf_idf_matrix.toarray()


tfidf_matrix = tf_idf_matrix

cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
print (cos_similarity_matrix)

related_docs_indices = cos_similarity_matrix.argsort()[:-5:-1]
related_docs_indices

cos_similarity_matrix[related_docs_indices]

raw_data.ix[0]
raw_data.ix[331]


# ---------------------------------------------------------
#  Silhouette Coefficient (Finding k)
# ---------------------------------------------------------

'''
If the ground truth labels are not known, evaluation must be performed
using the model itself. The Silhouette Coefficient
(sklearn.metrics.silhouette_score) is an example of such an evaluation,
where a higher Silhouette Coefficient score relates to a model
with better defined clusters. The Silhouette Coefficient
is defined for each sample and is composed of two scores:
a: The mean distance between sample & all other points in the same class.
b: The mean distance between sample & all other points in next nearest cluster

Silhouette coefficients (as these values are referred to as)
near +1 indicate that the sample is far away from the neighboring clusters.
A value of 0 indicates that the sample is on or very close to the decision
boundary between two neighboring clusters and negative values indicate that
those samples might have been assigned to the wrong cluster.

It is a combination measure assessing intra-cluster homogeneity
and inter-cluster separation. The best value is 1 and the worst value is -1.
Values near 0 indicate overlapping clusters.
'''
'''
for n_cluster in range(2, 40):
    kmeans = KMeans(n_clusters=n_cluster).fit(tf_idf_matrix)
    label = kmeans.labels_
    sil_coeff = silhouette_score(tf_idf_matrix, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(
            n_cluster, sil_coeff))
'''
# the silhouette coefficient quantifies the quality of clustering achieved so
# select the number of clusters that maximizes the silhouette coefficient.

# ---------------------------------------------------------
#  Gap Statistic (Finding k)
# ---------------------------------------------------------
# Tibshirani, Walther and Hastie in their 2001 paper
# Give the data to optimalK with maximum considered clusters of 15
# https://stats.stackexchange.com/questions/95290/how-should-i-interpret-gap-statistic
# https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/
# https://anaconda.org/milesgranger/gap-statistic/notebook
# https://github.com/milesgranger/gap_statistic/blob/master/Example.ipynb

'''
k, gapdf = optimalK(tf_idf_matrix, nrefs=5, maxClusters=25)
print("Optimal k is: ", k)

plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)
plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount,
            gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
plt.grid(True)
plt.xlabel('Cluster Count')
plt.ylabel('Gap Value')
plt.title('Gap Values by Cluster Count')
'''

# ---------------------------------------------------------
#  Calinski-Harabaz score (Finding k)
# ---------------------------------------------------------
# higher Calinski-Harabaz score relates a model with better defined clusters.
# The score is defined as the ratio between the within-cluster dispersion
# and the between-cluster dispersion.

# A sparse matrix was passed, but dense data is required.
'''
tf_idf_matrix_dense = tf_idf_matrix.toarray()

for n_cluster in range(2, 10):
    kmeans = KMeans(n_clusters=n_cluster).fit(tf_idf_matrix_dense)
    label = kmeans.labels_
    calinski_score = metrics.calinski_harabaz_score(tf_idf_matrix_dense, label)
    print("For n_clusters={}, The Calinski-Harabaz is {}".format(
            n_cluster, calinski_score))
'''
# ---------------------------------------------------------
#  Clustering text documents using k-means or minibatch kmeans
#
# ---------------------------------------------------------

num_clusters = 24  # this is optimal k

# =============================================================================
# clustering_model = KMeans(n_clusters=num_clusters,
#                           init='k-means++',
#                           random_state=10)
# =============================================================================

clustering_model = MiniBatchKMeans(n_clusters=num_clusters,
                                   init='k-means++',
                                   random_state=10,
                                   init_size=1000,
                                   batch_size=1000)

'''note below are labels in multidimensional space'''
labels = clustering_model.fit_predict(tf_idf_matrix.toarray())

# http://scikit-learn.org/stable/auto_examples/cluster/plot_mini_batch_kmeans.html
# Minibatch cluster centers error?
print("Top terms per cluster:")
order_centroids = clustering_model.cluster_centers_.argsort()[:, ::-1]
for i in range(num_clusters):
    print ("Cluster %d:" % i),
    for ind in order_centroids[i, :num_top_terms]:
        print (' %s' % terms[ind]),

# ---------------------------------------------------------
#  Evaluate Different Dimensionality Reduction Techniques
#  sometimes called vector compression technique
# ---------------------------------------------------------
# https://stats.stackexchange.com/questions/162532/evaluate-output-of-different-dimensionality-reduction-methods

# ---------------------------------------------------------
# PCA minimizes dimensions, preserving covariance of data.
# PCA uses singular value decomposition (SVD)
# PCA does not like sparce matrices..need dense thus .toarray()
# ---------------------------------------------------------

# Principle Component Analysis
start_time = time()
pca = PCA(n_components=2).fit(tf_idf_matrix.toarray())
pca_data2D = pca.transform(tf_idf_matrix.toarray())
pca_centers2D = pca.transform(clustering_model.cluster_centers_)
end_time = time()

# Truncated SVD - SVD KMeans..singular value decomposition
start_time = time()
svd = TruncatedSVD(n_components=2)
svd_data2D = svd.fit_transform(tf_idf_matrix.toarray())
end_time = time()
svd_centers2D = svd.transform(clustering_model.cluster_centers_)

# t-SNE plot..t-distributed Stochastic Neighbor Embedding
# https://stats.stackexchange.com/questions/263539/k-means-clustering-on-the-output-of-t-sne
# https://stackoverflow.com/questions/37932928/perform-clustering-using-t-sne-dimensionality-reduction
start_time = time()
tSNE = TSNE(n_components=2)
tSNE_data2D = tSNE.fit_transform(tf_idf_matrix.toarray())
end_time = time()
# tSNE_centers2D = tSNE.transform(clustering_model.cluster_centers_)


# ---------------------------------------------------------
#  Visualization of Clusters Techniques
# ---------------------------------------------------------

'''note the labels below can be derived from the reduced or not reduced data'''
'''labels = algorithm(*args, **kwds).fit_predict(tf_idf_matrix.toarray())'''
'''There is also a possible number of colors limit going below'''

def plot_clusters(data, dim, algorithm, args, kwds):
    sns.set_context('poster')
    sns.set_color_codes()
    plot_kwds = {'alpha': 0.25, 's': 80, 'linewidths': 0}
    start_time = time()
    clustering_model = algorithm(*args, **kwds)
    labels = clustering_model.fit_predict(data)
    end_time = time()
    centers = dim # need to splice input here pca, tSNE, SVD
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
   # plt.scatter(centers.T[0], centers.T[1], marker='o', s=200, linewidths=3, c='black', alpha=0.25)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.text(0.20, 0.37, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
    # plt.xlabel("Feature space for the 1st feature")
    # plt.ylabel("Feature space for the 2nd feature")

'''
------- plot_clusters function inputs -----------
data = has to be 2D data
algorithm = k-Means type with parameters including "k"
dim = is the dimension reduction technique
'''
plot_clusters(pca_data2D, pca_centers2D, cluster.KMeans, (
        ), {
    'init': 'k-means++',
    'random_state': 10,
    'n_clusters': 24})

plot_clusters(pca_data2D, pca, cluster.MiniBatchKMeans, (
        ), {
    'init': 'k-means++',
    'random_state': 10,
    'n_clusters': 24})

plot_clusters(svd_data2D, svd, cluster.KMeans, (), {'n_clusters': 24})
plot_clusters(tSNE_data2D, tSNE, cluster.KMeans, (), {'n_clusters': 24})

metrics.silhouette_score(pca_data2D, labels, sample_size=1000, metric='euclidean')
metrics.silhouette_score(tSNE_data2D, labels, sample_size=1000, metric='euclidean')
metrics.silhouette_score(svd_data2D, labels, sample_size=1000, metric='euclidean')

print("Top terms per cluster:")
order_centroids = clustering_model.cluster_centers_.argsort()[:, ::-1]
for i in range(num_clusters):
    print ("Cluster %d:" % i),
    for ind in order_centroids[i, :num_top_terms]:
        print (' %s' % terms[ind]),

'''
-----Interactive Bokeh plots of the data
Currently, the main function is that you are able to show 
the title and abstract information of selected data points.
You can add codes to update what to show in the show_info() function.

You need to run this in anaconda prompt by the command:
    >> bokeh serve Patent_analysis_Revision_1.py
Navigate to the URL
http://localhost:5006/selection_histogram
on your server.
You will probably need to wait a few minutes and see the graphs produced by the 
previous sections of the codes popping out on new windows. Finally the Bokeh 
interactive plot will show up. 
'''        
        
import bokeh as bokeh
import pandas as pd
from bokeh.plotting import figure
from bokeh.layouts import layout, widgetbox
from bokeh.models import ColumnDataSource, HoverTool, BoxZoomTool, ResetTool, PanTool
from bokeh.models.widgets import Slider, Select, TextInput, Div
from bokeh.models import WheelZoomTool, SaveTool, LassoSelectTool
from bokeh.io import curdoc
from bokeh.io import output_notebook, push_notebook
from bokeh.io import output_file, show

# Below is setting up the colors for the cluster labels to be shown in the Bokeh interactice plot
from bokeh.palettes import viridis
colors_uniq=viridis(len(np.unique(labels)))
my_colormap = dict(zip(np.unique(labels), colors_uniq))
colors=list(map(my_colormap.get, labels))

# Below is to give data to the bokeh data model

#data_dict=dict(x=pca_data2D[:,0], y=pca_data2D[:,1], index=list(range(0,len(pca_data2D))), detail=corpus, 
#               labels=labels, colors=colors)
source = ColumnDataSource(data=dict(x=pca_data2D[:,0], y=pca_data2D[:,1], index=list(range(0,len(pca_data2D))), 
                                    detail=corpus, labels=labels, colors=colors))

# Below is to define tools to use for interaction.
# HoverTool can be added with title, author, etc. But I think they should be first included in source
hover = HoverTool(tooltips=[
    ("index", "$index"),
    ("(x,y)", "($x, $y)")])

TOOLS = [hover, BoxZoomTool(), LassoSelectTool(), WheelZoomTool(), PanTool(),
    ResetTool(), SaveTool()]

# Below is to create the figure
p = figure(
    plot_height=600,
    plot_width=700,
    title="Trial plot",
    tools=TOOLS,
    x_axis_label="x_PCA2D",
    y_axis_label="y_PCA2D",
    toolbar_location="above")

p.circle(
    y="y",
    x="x",
    source=source,
    color='colors',
    size=7,
    alpha=0.4)

# Below is to create the widget to show whatever you want for the selected patents 
# by LassoSelectTool
details = Div(text="Selection Details:", width=800)


# Function that will show word cloud of selected documents upon change. This function is 
# not finished yet, because the wordcloud generated image cannot be shown in Bokeh plot so far
def word_cloud_gen(attr, old, new):

    corpus_clean = [clean(doc).split() for doc in corpus]  # clean the corpus
    corpus_uni = byteify(corpus_clean)  # remove unicode encoding for wordgram

    # ---------------------------------------------------------
    #  Tokenize words
    # ---------------------------------------------------------
    tokens = tokenizer(corpus_uni)

    # ---------------------------------------------------------
    #  Frequency analysis
    # ---------------------------------------------------------
    fd = FreqDist(tokens)
    fd.most_common(40)
    ##fd.plot(20)
    ##fd.plot(20, cumulative=True)  # adds cummulative count from left to right

    # generate a word cloud based on frequency
    tokens_str = str(tokens)
    tokens_str = tokens_str.replace("'", "")

    # http://amueller.github.io/word_cloud/
    wordcloud = WordCloud(max_font_size=40).generate(tokens_str)

    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    

# Function to show text details of the selected patents
# You can add some analysis including machine learning studies into this function
# But still, like the previous function, you need to figure out how to show images
# if you need images to tell the analysis results. You can probably let a new window 
# pop up and show it. But that is quite un-professional.
# The best way is to link the main Bokeh plot (that you do selection on to another bokeh plot 
# that shows the analysis for the selected data     
def show_info(attr, old, new):
    """ Function will be called when the poly select (or other selection tool)
    is used. Determine which items are selected and show the details below
    the graph
    """
    selected = source.selected["1d"]["indices"]
    if selected:
        data = df_dropna.iloc[selected, 0:3]
        temp = data.T
        details.text = temp.style.render()
    else:
        details.text = "Selection Details"

# choose what function to apply to the selected patents
source.on_change("selected", show_info)

# Create the figure and widget layout in order to show the graph
l = layout([[p], [details]], sizing_mode="fixed")  #I figured that if you rerun this line and curdoc().add_root(l) will give error. Not sure why.
curdoc().add_root(l)
curdoc().title = "Patent analysis"






