import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import datasets
from sklearn import svm
from pony.orm import *
from wordcloud import WordCloud, STOPWORDS
import pickle
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import collections
from stemming.porter2 import stem

# handles csv correlating clusters to publication_id's

df = pd.read_csv("pubmedID_min5clusters_v2.csv", names=['PubID','ClusterNo'])
organized_by_cluster = []


def parse_clustered_publications():
    """
    Stores and dumps a dictionary of all publications to their respective \
    clusters
    """
    org_list = []
    # dict ClusterNo : {PubID-1, PubID-2, ... , PubID-n}
    org_dict = dict()
    for j in range(df["ClusterNo"].min(), df['ClusterNo'].max()):
        org_list.clear()
        org_list.append(df.loc[df['ClusterNo'] == j]['PubID'].values.tolist())
        org_dict[j] = org_list[0]
        # organized_by_cluster.append(temp_list)
    pickle.dump(org_dict, open("organized_dict.pickle", "wb"))


def get_pickle(filename):
    return pickle.load(open(filename, "rb"))

# parse_clustered_publications()
organized_by_cluster = get_pickle("organized_dict.pickle")


# handles database with words and papers

db_word = Database()

class Words(db_word.Entity):
    _table_ = "words"
    rowid = PrimaryKey(int)
    word = Required(str)

class PaperWordCount(db_word.Entity):
    _table_ = "paper_word_count"
    paper = PrimaryKey(int) # PMCID as integer
    word = Required(int)
    count = Required(int)

db_word.bind('sqlite', "wordfreq.sqlite")
db_word.generate_mapping()

db_paper = Database()

class Papers(db_paper.Entity):
    _table_ = "papers"
    id = PrimaryKey(int) # PMCID as integer
    title = Required(str)
    abstract = Required(str)
    keywords = Required(str)
    year = Required(int)
    month = Required(int)

class Citations(db_paper.Entity):
    _table_ = "paper_citation"
    paper = PrimaryKey(int) # PMCID as integer
    cited_by = Required(int) # PMCID as integer

db_paper.bind('sqlite', 'PaperNetworkSqlite.alzheimer.db')
db_paper.generate_mapping()


def nlp(words):
    """
    Inputs a list of words and returns a list of filtered words
    """
    filtered = []

    stopWords = set(stopwords.words('english'))
    for w in words:
        if w.lower() not in stopWords:
            x = w.lower()
            x = x.replace("’", "'")
            x = x.replace("'s", "")
            x = x.replace(":", "")
            if nltk.stem.WordNetLemmatizer().lemmatize(x, 'v') == x:
                x = nltk.stem.WordNetLemmatizer().lemmatize(x, 'n')
            else:
                x = nltk.stem.WordNetLemmatizer().lemmatize(x, 'v')
            filtered.append(x)
    return filtered

# finds most common element in a list
def most_common(lst, ct=5):
    counter = collections.Counter(lst)
    print(counter.most_common(ct), "\n")

def show_wordcloud(lst, subset="all", feature="keywords"):
    """
    Displays a wordcloud given a list of words
    """
    all_string = ' '.join(map(str, lst))
    wordcloud = WordCloud(background_color='white',
                          width=2400,
                          height=1500
                          ).generate(all_string)
    plt.imshow(wordcloud)
    plt.axis('off')
    if subset == "all":
        plt.title("Most common "+feature+" in the entire database")
    else:
        plt.title("Most common "+feature+" in cluster: "+subset)
    plt.show()

# given a cluster number, can we find the keywords in the papers in the cluster
def find_keywords(cluster_number=0, all_=False):
    """
    Returns one value:
        1. A list of all keywords
    """
    all_words = []

    # if all keywords are desired
    if all_ == True:
        with db_session:
            for k in (select(p.keywords for p in Papers if p.keywords != None)):
                all_words += k.split(",")

        return all_words

    paper_list = organized_by_cluster[cluster_number]
    for paper_id in paper_list:
        with db_session:
            # gets a list of all keywords given a paper_id
            temp = select(p.keywords for p in Papers if p.id == paper_id and p.keywords != None)[:]
            if temp:
                # separates the concatanated string into individual list elements by commas
                temp = temp[0].split(",")
            # adds temp to all_words
            all_words += temp

    # returns all words in a list format and in a concatanated string
    return all_words

def test_find_keywords():
    # get all keywords
    all_keywords = find_keywords(all_=True)
    # get most common keyword in database
    most_common(all_keywords, 3)
    # display wordcloud of all keywords
    show_wordcloud(all_keywords)

    # get list of keywords for cluster 66
    keywords_66_list = find_keywords(66)
    # get most common keyword in cluster 66
    most_common(keywords_66_list, 3)
    # display wordcloud of keywords in cluster 66
    show_wordcloud(keywords_66_list, "66")

def find_titles(cluster_number=0, all_=False):
    """
    Returns one value:
        1. Filtered list of words
    """
    all_titles = []
    all_title_words = []

    # if looking at all cluster information
    if all_ == True:
        with db_session:
            all_titles = select(p.title for p in Papers if p.title != None or p.title != "")[:]
            for t in all_titles:
                all_title_words += t.split(' ')

        # Natural Language Processing + Return
        return nlp(all_title_words)

    # if looking at a certain cluster
    paper_list = organized_by_cluster[cluster_number]
    for paper_id in paper_list:
        with db_session:
            # gets the title of a given a paper_id
            temp_title = select(p.title for p in Papers if p.id == paper_id and p.title != None and p.title!="")[:]
            all_titles += temp_title

    for t in all_titles:
        all_title_words += t.split(' ')

    # Natural Language Processing + Return
    return nlp(all_title_words)

def test_find_titles():
    # For entire dataset
    # Get a list of words in titles (processed)
    all_t_w_p = find_titles(all_=True)
    # most common words in titles
    most_common(all_t_w_p, 7)
    # shows wordcloud
    show_wordcloud(all_t_w_p, feature="words in titles")

    # For Cluster 66
    # Get a list of words in titles in Cluster 66 (processed)
    all_t_w_p_66 = find_titles(66)
    # most common words in cluster 66 titles
    most_common(all_t_w_p_66, 7)
    # shows wordcloud
    show_wordcloud(all_t_w_p_66, feature="words in titles", subset="66")
