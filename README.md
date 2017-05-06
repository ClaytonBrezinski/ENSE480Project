# ENSE480Project
This is a sentiment analysis tool that takes a range of review data sets from sources such as Amazon product reviews, IGN 
game reviews, and IMDB movie reviews, and then builds one of 3 user selected analyzer models. Once a model is 
constructed, the system preforms sentiment analysis on a given test data and outputs the results into a .TSV file with 
the model's name. 

## Dependencies
##### Automatically Installed:
* [Python](https://www.python.org/) 3.6.x
* [Pandas](http://pandas.pydata.org/) 0.19.2
* [sklearn](http://scikit-learn.org/) 0.0
* [numpy](http://www.numpy.org/) 1.12.0
* [BeautifulSoup4](https://pypi.python.org/pypi/beautifulsoup4) 4.4.1
* [scipy](https://www.scipy.org/) 0.19.0
* [gensim](https://radimrehurek.com/gensim/) 1.0.1
* [NLTK](http://www.nltk.org/) 3.2.2

This can be done by opening the console/command window on your machine and entering: 
```python
import nltk
nltk.download()
```
A UI will appear, you will then pressing the download button to download everything that the Natural Language Toolkit 
has to offer.

## Supported Models
All models are held under a Random Forest Classifier as features
* Bag of Words
* Word to Vector - Average
* Word to Vector - Centroid