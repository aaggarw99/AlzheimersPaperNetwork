import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import datasets
from sklearn import svm
from pony.orm import *
from wordcloud import WordCloud, STOPWORDS


# df = pd.read_csv("pubmedID_min5clusters_v2.csv", names=['PubID','ClusterNo'])
# print(df.head())

# df.rename(columns={"ClusterNo":"LabelInt"}, inplace=True)

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
