import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import datasets
from sklearn import svm
from pony.orm import *
from wordcloud import WordCloud, STOPWORDS
import pickle

# handles csv correlating clusters to publication_id's

df = pd.read_csv("pubmedID_min5clusters_v2.csv", names=['PubID','ClusterNo'])
organized_by_cluster = []


def parse_clustered_publications():
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

parse_clustered_publications()
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

def show_wordcloud():
    all_words = []

    with db_session:
        for k in (select(p.keywords for p in Papers if p.keywords != None)):
            all_words += k.split(",")

    all_str = ' '.join(map(str, all_words))

    wordcloud = WordCloud(background_color='white',
                          width=2400,
                          height=1500
                          ).generate(all_str)

    plt.imshow(wordcloud)
    plt.axis('off')
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
