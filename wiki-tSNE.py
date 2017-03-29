
# coding: utf-8

# This notebook will show you how to organize in 2D a set of documents/articles/posts so that articles with similar content are grouped near to each other. The example I am using is a set of Wikipedia articles of [Political ideologies](https://en.wikipedia.org/wiki/List_of_political_ideologies), but in principle it can be used for any set of documents. 
# 
# The result of this notebook [can be viewed live here](https://www.genekogan.com/works/wiki-tSNE/).
# 
# ### Procedure
# 
# The pipeline consists of two steps.
# 
# 1) Convert all of the articles to a [tf-idf matrix](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).
# 
# tf-idf stands for "term frequency inverse document frequency" and is commonly used in natural language processing applications dealing with large collections of documents. A tf-idf matrix is an $n * m$ sparse matrix consisting of $n$ rows, corresponding to our $n$ documents, and $m$ columns, corresponding to the $m$ unique "terms" (usually just words but can be n-grams or other kinds of tokens) that appear in the entire corpus.
# 
# Each entry in the matrix, $tfidf(i,j)$ can be interpreted as the "relative importance" of term $j$ to document $i$.  It is calculated as
# 
# $$tfidf(i,j) = tf(i,j)*idf(i,j)$$
# 
# $tf(i, j)$ is the "term frequency," i.e. the percentage of terms in document $i$ which are term $j$. For example, in the document "the cat in the hat", the term "the" has a $tf$ of (2 / 5) = 0.4. Thus $tf$ is high when the term is frequently found in the document.
# 
# $idf(i, j)$, not to be confused with [this IDF](https://twitter.com/idfspokesperson/status/547144026445471744) is the "inverse document frequency." It is given by:
# 
# $$idf(i, j) = log(\frac{N}{n_j})$$
# 
# where $N$ is the number of documents in the corpus and $n_j$ is the number of documents which contain term $j$. When $n_j$ is high, this value shrinks towards 0. This happens when the term frequently appears in many or all documents, thus common terms like "the", "a", "it", etc will have a low $idf$ score because they appear in most documents. Conversely, when the term rarely appears in documents ($n_j$ is low), then $idf$ score will be high. These tend to be special or topic-specific words which appear in few of the documents.
# 
# So intuitively, the $tfidf$ score for a term in a document goes higher if the term appears frequently in the document and appears infrequently in other documents (so that term is important to that document).
# 
# 2) This gives you a high-dimensional matrix of n documents which can be reduced to 2 dimensions using the [t-SNE](https://lvdmaaten.github.io/tsne/) dimensionality reduction technique. A better description of how t-SNE works can be found in the link.
# 
# ### Installation
# 
# You need [nltk](http://www.nltk.org/install.html) and [scikit-learn](http://scikit-learn.org/) to run most of this notebook.
# 
#     pip install -U nltk
#     pip install -U scikit-learn
# 
# Also, if you don't already have [numpy](http://www.numpy.org/), you should install it as well (it's only used to normalize the data later). 
# 
#     pip install numpy
# 
# Additionally, if you are following this example and wish to extract articles from Wikipedia, you need the python [wikipedia](https://pypi.python.org/pypi/wikipedia/) library. 
# 
#     pip install wikipedia
#   

# First make sure all these imports work (minus wikipedia if you intend to use another corpus). We will assume wikipedia from here.

# In[2]:

import string
import time
import pickle
import json
import sys
import wikipedia
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD


# In this example, we are going to cluster all of the links found in the Wikipedia page "[List of political ideologies](https://en.wikipedia.org/wiki/List_of_political_ideologies)." First, we will open the page in main, then add all of the raw text of the articles into a dictionary called token_dict.

# In[4]:
    
# read game result dump from TheWikiGame
print('reading game dump')
snapshot = open('snapshot.txt')
line_nmbrs = [56, 76, 90]
lines = snapshot.readlines()
mylist = set()
paths = [lines[i-1].split(' â†’ ') for i in line_nmbrs]
for i in line_nmbrs:
    for word in lines[i-1].split(' â†’ '):    # right arrow is converted to this gibberish
        mylist.add(word)
mylist = list(mylist)
    
#main = wikipedia.page('List of political ideologies')  # replaced main.links with mylist
token_dict = {}
for i, article in enumerate(mylist):
    if article not in token_dict:
        time.sleep(5)  # helps to avoid hangups. Ctrl-C in case you get stuck on one
        try:
            print("getting article %d/%d: %s" % (i+1, len(mylist), article))
            text = wikipedia.page(article)
            print("  loading content of article: %s" % article)
            token_dict[article] = text.content
        except Exception as e:
            print(" ==> error processing " + article + ": " + str(e))


# You may find it useful to save the dictionary so as to not have to re-download all the articles later.

# In[3]:

#pickle.dump(token_dict, open("token_dict_political_ideologies.p", "wb" ))

# later you can retrieve it like this:
#token_dict = pickle.load(open("token_dict_political_ideologies.p", "r" ))


# Next, we will evaluate the tf-idf matrix of our collection of articles. First we need to define a tokenizer function which will properly convert all the raw text of the articles into a vector of tokens (our terms).
# 
# The function `tokenize` will take the raw text, convert it to lower-case, strip out punctuation and special characters, remove all stop words (common words in english, e.g. "the", "and", "i", etc), stem the remaining words ([using Porter stemmer](https://en.wikipedia.org/wiki/Stemming)) to unify tokens in different parts-of-speech (e.g. "run", "running", "ran" --> "run"), and output what's left as a vector.
# 
# From there we run the tfidf vectorizer which will return us the resulting tfidf matrix. Then we use [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition) to reduce the dimensionality before t-SNE (this is optional, but some sources recommend it). 
# 
# Note, calculating the tfidf matrix can take a while.

# In[4]:

def tokenize(text):
    text = text.lower() # lower case
    for e in set(string.punctuation+'\n'+'\t'): # remove punctuation and line breaks/tabs
        text = text.replace(e, ' ')	
    for i in range(0,10):	# remove double spaces
        text = text.replace('  ', ' ')
    text = text.translate(string.punctuation)  # punctuation
    tokens = nltk.word_tokenize(text)
    text = [w for w in tokens if not w in stopwords.words('english')] # stopwords
    stems = []
    for item in tokens: # stem
        stems.append(PorterStemmer().stem(item))
    return stems

# calculate tfidf (might take a while)
print("calculating tf-idf")
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(token_dict.values())
dim_red = int(2.0/3.3*len(mylist))
print("reducing tf-idf to %d dim" % dim_red)
tfs_reduced = TruncatedSVD(n_components=dim_red, random_state=0).fit_transform(tfs)
print("done")


# Let's quickly see what we have computed. Your exact results may vary.  
# 
# `tfs` is our raw tf-idf matrix.  We can print it and see the values. For example, tfs(0, 20000) is the tfidf score for document 0, term 20000. We can query to find out what this term is. 

# In[9]:

#print(tfs)
#print("term 7448 = \"%s\""%tfidf.get_feature_names()[7448])


# Let's also look at tfs_reduced, which comes from our truncated SVD. note, we requested 500 dims, but it returned 408 because that's how many documents we managed to download.

# In[10]:

#print("size of TSVD: "+str(tfs_reduced.shape))
#print(tfs_reduced)


# Finally, we run t-SNE on the result, normalize the t-SNE results (for convenience), and save the final coordinates to a json file.
# 
# Notice that we run it on `tfs_reduced`. You can also run it on the original tf-idf matrix (results should be similar). Make sure to remember to make it a dense matrix, i.e. `tfs.todense()`.
# 
# You may also want to play a bit with the parameters, like the size of the SVD reduction (probably minor effect) or the perplexity of the t-SNE.
# 
# At this point, the 2d normalized t-SNE points have been saved to a JSON file, along with the corresponding names of the articles. A nice way to visualize it is to display the terms in the browser. I've made a p5.js sketch which does this.

# In[11]:

#model = TSNE(n_components=2, perplexity=50, verbose=2).fit_transform(tfs.todense())
print('running T-SNE')
model = TSNE(n_components=2, perplexity=3, verbose=2, angle=0.000001).fit_transform(tfs_reduced)

# save to json file
x_axis=model[:,0]
y_axis=model[:,1]
x_norm = (x_axis-np.min(x_axis)) / (np.max(x_axis) - np.min(x_axis))
y_norm = (y_axis-np.min(y_axis)) / (np.max(y_axis) - np.min(y_axis))
x_scaled = x_norm * 1800 + 60
y_scaled = y_norm * 800 + 90

print('exporting json file for display in browser')
# construct path coordinates
paths_coords = []
for path in paths:
    paths_coords.append([(x_scaled[mylist.index(waypoint)],y_scaled[mylist.index(waypoint)]) for waypoint in path])

data = {"x":x_scaled.tolist(), "y":y_scaled.tolist(), "names":list(token_dict.keys()), "paths":paths_coords}
with open('visualize/data_adjusted.json', 'w') as outfile:
    json.dump(data, outfile)


# We now have put the t-SNE normalized coordinates and document names into the file "data_political_ideologies.json" so they can be used in other environments. This codebase comes with an example (in the `visualize` folder) of displaying these in a webpage using [p5.js](http://www.p5js.org) (javascript). This script uses a simple collision detection procedure to space out all the text so they are not overlapping. 
