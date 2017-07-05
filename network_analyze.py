import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import datasets
from sklearn import svm
from pony.orm import *
from wordcloud import WordCloud, STOPWORDS
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import collections


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

# finds most common element in a list
def most_common(lst, ct=5):
    counter = collections.Counter(lst)
    print(counter.most_common(ct), "\n")

def show_wordcloud(all_string, subset="all", feature="keywords"):
    """
    Displays a wordcloud given a concatanated string separating words with spaces
    """
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



# querying
# with db_session:
    # select(count(w.word) for w in Words)[:].show()
    # select(count(p.id) for p in Papers)[:].show()
    # string = select((p.id, p.keywords, p.month) for p in Papers if p.year == 2017 and p.month == 2)
    # x = select(p.keywords for p in Papers if p.id == 5218855)
    # x = Papers.get(id=5218855)
    # y = Papers.get(id=5325057)
    #
    # keywordsx = x.keywords.split(",")[:]
    # keywordsy = y.keywords.split(",")[:]

    # Papers.select_by_sql("SELECT keywords FROM Papers GROUP BY keywords ORDER BY COUNT(*) DESC LIMIT 1").show()

# given a cluster number, can we find the keywords in the papers in the cluster
def find_keywords(cluster_number=0, all_=False):
    """
    Returns two values:
        - A list of all keywords
        - A concatanated string of all keywords
    """
    all_words = []

    # if all keywords are desired
    if all_ == True:
        with db_session:
            for k in (select(p.keywords for p in Papers if p.keywords != None)):
                all_words += k.split(",")

        all_str = ' '.join(map(str, all_words))

        return all_words, all_str

    # els

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

    # creates a string of all keywords (used for wordcloud)
    all_str = ' '.join(map(str, all_words))

    # returns all words in a list format and in a concatanated string
    return all_words, all_str

def test_find_keywords():
    # get all keywords
    all_keywords = find_keywords(all_=True)[1]
    # display wordcloud of all keywords
    show_wordcloud(all_keywords)

    # get list of keywords for cluster 66
    keywords_66_list = find_keywords(66)[0]
    # get concatanated string of keywords for cluster 66
    keywords_66_string = find_keywords(66)[1]
    print(most_common(keywords_66_list))
    # display wordcloud of keywords in cluster 66
    show_wordcloud(keywords_66_string, "66")

# test_find_keywords()

def find_titles(cluster_number=0, all_=False):
    """
    Returns three values:
        1. List of unprocessed title strings
        2. String of unprocessed title strings
        3. List of processed title words
        4. String of processed title words
    """
    all_titles = []
    all_title_str = ""

    all_title_words = []

    all_title_words_pro = []
    all_title_words_pro_str = ""

    # if looking at all cluster information
    if all_ == True:
        with db_session:
            all_titles = select(p.title for p in Papers if p.title != None or p.title != "")[:]
            all_titles = [x.lower() for x in all_titles]
            all_title_str = ' '.join(map(str, all_titles))
            for t in all_titles:
                all_title_words += t.split(' ')
            stopWords = set(stopwords.words('english'))

            for w in all_title_words:
                if w not in stopWords:
                    all_title_words_pro.append(w)
            all_title_words_pro_str = ' '.join(map(str, all_title_words_pro))

        return all_titles, all_title_str, all_title_words_pro, all_title_words_pro_str

    # if looking at a certain cluster
    paper_list = organized_by_cluster[cluster_number]
    for paper_id in paper_list:
        with db_session:
            # gets the title of a given a paper_id
            temp_title = select(p.title for p in Papers if p.id == paper_id and p.title != None and p.title!="")[:]
            all_titles += temp_title

    # filtering for stopwaords
    for t in all_titles:
        all_title_words += t.split(' ')
    stopWords = set(stopwords.words('english'))

    for w in all_title_words:
        if w.lower() not in stopWords:
            all_title_words_pro.append(w.lower())

    # concatanate
    all_title_str = ' '.join(map(str, all_titles))

    # concatanate
    all_title_words_pro_str = ' '.join(map(str, all_title_words_pro))

    # returns all words in a list format and in a concatanated string
    return all_titles, all_title_str, all_title_words_pro, all_title_words_pro_str


def test_find_titles():
    # For entire dataset
    all_t, all_t_str, all_t_w_p, all_t_w_p_str = find_titles(all_=True)

    # Unprocessed
    most_common(all_t, 7) # most common title
    # show_wordcloud(all_t_str, feature="titles")

    # Processed
    most_common(all_t_w_p, 7) # most common words in titles
    # show_wordcloud(all_t_w_p_str, feature="words in titles")

    # For Cluster 66
    all_t_66, all_t_str_66, all_t_w_p_66, all_t_w_p_str_66 = find_titles(66)

    # Unprocessed
    most_common(all_t_66, 7) # most common title
    # show_wordcloud(all_t_str_66, feature="titles", subset="66")

    # Processed
    most_common(all_t_w_p_66, 7)
    # show_wordcloud(all_t_w_p_str_66, feature="words in titles", subset="66")

# test_find_titles()
