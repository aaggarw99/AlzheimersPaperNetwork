import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import datasets
from sklearn import svm
from pony.orm import *


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
with db_session:
    select(count(w.word) for w in Words)[:].show()
    select(count(p.id) for p in Papers)[:].show()
