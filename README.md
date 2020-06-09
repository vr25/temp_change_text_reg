Textual Similarity on MD&A disclosure
===============

Code and data for "Textual Similarity on MD&A disclosure"

## Prerequisites
This code is written in python. To use it you will need:
- Python 3.6
- A recent version of [scikit-learn](https://scikit-learn.org/)
- A recent version of [Numpy](http://www.numpy.org)
- A recent version of [NLTK](http://www.nltk.org)
- A recent version of [gensim](https://radimrehurek.com/gensim/)

## Getting started
We provide all the similarity scores for the different methods described in the paper along with the data statiscs.

To create the doc2vec model and then use it to find the similar document vectors:
run ```python3 compute_doc2vec_sim.py``` 
