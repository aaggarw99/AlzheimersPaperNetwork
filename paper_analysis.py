from database import *
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from matplotlib import style
import collections
from stemming.porter2 import stem
import re
import numpy as np
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder
import itertools
from collections import Counter
import pandas as pd
import os
import pickle
import json

class PaperAnalysis():

    def __init__(self):
        df = pd.read_csv("pubmedID_min5clusters_v2.csv", names=['PubID', 'CusterNo'])
        self.min5_ids = df.PubID
        if not os.path.exists('paper_data.pickle'):
            self.data = self.get_valid_set()
            pickle.dump(self.data, open('paper_data.pickle', 'wb'))
        else:
            with open('paper_data.pickle', 'rb') as f:
                with db_session:
                    self.data = pickle.load(f)

    def get_paper_from_id(self, idx, with_a_k=False):
        """
        Args:
        -----
        idx = id of paper
        with_a_k = when set to true, will only return paper information if it has an abstract and keywords

        Return:
        --------
        Paper object attributes in the following format:
        Paper properties: [id, title, abstract, keywords, year, month]

        """
        with db_session:
            if with_a_k==False:
                if Papers.get(id=idx):
                    if Papers.get(id=idx).abstract != None and Papers.get(id=idx).keywords != None:
                        return Papers.get(id=idx)
                print("No paper with an abstract or keywords was found")
                return []
            else:
                if Papers.get(id=idx):
                    return Papers.get(id=idx)
                print("No paper was found")
                return []
                

                
    def get_keywords(self, idx):
        """
        Args:
        -----
        idx : id of the paper (-1 indicates the entire database)
        
        Return:
        -------
        Return list of keywords
        """
        all_keywords = []
        with db_session:
            # look at the entire database keywords
            if idx == -1:
                for k in (select(p.keywords for p in Papers if p.keywords != None)):
                    all_keywords += k.split(',')

                return all_keywords
            if Papers.get(id=idx):
                
                keyword_str = Papers.get(id=idx).keywords
                if keyword_str == None:
                    return []
                all_keywords += keyword_str.split(",")
                return all_keywords
            return []

    def get_title(self, idx):
        """
        Args:
        -----
        idx : id of the paper (-1 indicates the entire database)
        
        Return:
        -------
        A list of words in the title of paper idx
        """
        all_title_words = []

        with db_session:

            if idx == -1:
                all_titles = select(p.title for p in Papers if p.title != None or p.title!="")[:]
                for t in all_titles:
                    all_title_words += t.split(' ')

                return all_title_words

            if Papers.get(id=idx):
                title_str = Papers.get(id=idx).title
                if title_str == None:
                    return []
                all_title_words += title_str.split(" ")
                return all_title_words
            return []
            
    def get_abstract(self, idx):
        """
        Args:
        -----
        idx : id of the paper
        
        Return:
        -------
        Return a list of words in the title of paper idx
        """
        all_abstract_words = []

        with db_session:
            if idx == -1:
                all_abstract = select(p.abstract for p in Papers if p.abstract != None and p.abstract!="")[:]
                for a in all_abstract:
                    all_abstract_words += a.split(' ')
                return all_abstract_words

            if Papers.get(id=idx):
                abstract_str = Papers.get(id=idx).abstract
                if abstract_str == None:
                    return []
                all_abstract_words += abstract_str.split(" ")
                return all_abstract_words

            return []

    def get_citations(self, idx):
        with db_session:
            return Citations.get(paper=idx).cited_by
                
    def keyword_in_abstract(self, idx):
        """
        Args:
        -----
        idx: paperID

        Return:
        -------
        Percentage of how often keywords appear in the abstract OR -1 if the paper has either no keywords or abstract
        """

        # removes duplicates in keywords list and abstract words list

        unique_k = list(set(self.nlp(self.get_keywords(idx))))
        split_k = []
        for i in unique_k:
            split_k += i.split(" ")
        if not unique_k:
            return -1
        unique_a = list(set(self.nlp(self.get_abstract(idx))))
        if not unique_a:
            return -1
        in_abstract = 0
        for i in split_k:
            if i in unique_a:
                in_abstract += 1
                continue

        return (in_abstract / len(split_k))



    
    # Natural Word Processing
    def nlp(self, words):
        """
        Performs natural language processing on a list of words

        Args:
        -----
        words : list of words

        Return:
        -------
        List of filtered words
        """
        filtered = []

        stopWords = set(stopwords.words('english'))
        words = [w.lower() for w in words if w.lower() not in stopWords]
        words = [w for w in words if re.match("[a-zA-Z]{2,}", w)]
        words = [re.sub(r'[^\w\s]','',w) for w in words]
        for w in words:
            x = w
            x = x.replace("’", "'")
            x = x.replace("'s", "")
            x = x.replace(":", "")
            if nltk.stem.WordNetLemmatizer().lemmatize(x, 'v') == x:
                x = nltk.stem.WordNetLemmatizer().lemmatize(x, 'n')
            else:
                x = nltk.stem.WordNetLemmatizer().lemmatize(x, 'v')
            if w == 'alzheimer disease':
                x = 'alzheimers disease'
            filtered.append(x)
        return filtered

    # finds most common element in a list
    def most_common(self, lst, ct=5):
        """
        Args:
        -----
        lst : list of items
        ct : integer
        Specifies to print the top "ct" items
        
        Return:
        -------
        The most common items in a list
        """
        counter = collections.Counter(lst)
        return counter.most_common(ct)

    def show_wordcloud(self, lst, subset="all", feature="keywords"):
        """
        Displays a wordcloud

        Args:
        -----
        lst : list of words
        subset : specify "all" or cluster_number for the visual
        feature : specify the type of Paper attribute for the visual
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

    def ngram_analyze(self, lst, model="student_t"):
        """
        Documentation for analysis tools:
        http://www.nltk.org/_modules/nltk/metrics/association.html

        Uses student_t distribution to analyze a list of words by splitting them into \
        tuples of 3 elements: eg. (a, b, c), (b, c, d), ...

        The distribution assigns a score to each tuple. This function returns the \
        highest score words

        Args:
        -----
        lst : a list of words
        model : the chosen model for ngram analysis (student_t, chi_sq, mi_like, pmi, jaccard)
        """
        lst = self.nlp(lst)
        string = " ".join(map(str, lst))
        words = nltk.word_tokenize(string)

        measures = TrigramAssocMeasures()

        finder = TrigramCollocationFinder.from_words(words)

        scores = []

        if model == "student_t":
            scores = finder.score_ngrams(measures.student_t)[:]
        elif model == "chi_sq":
            scores = finder.score_ngrams(measures.chi_sq)[:]
        elif model == "mi_like":
            scores = finder.score_ngrams(measures.mi_like)[:]
        elif model == "pmi":
            scores = finder.score_ngrams(measures.pmi)[:]
        elif model == "jaccard":
            scores = finder.score_ngrams(measures.jaccard)[:]
        else:
            print("Not valid model!")

        scores.sort(key=lambda i:i[1], reverse=True)
        top = scores[:3]
        return top
    # LDA model

    def categorize(self):
        """
        Categorize based on keywords
        """
        summ = 0
        num = 0
        
        cnt = Counter()
        for paper in self.data:
            if paper.keywords:
                keywords = paper.keywords.split(',')
                summ += len(keywords)
                num += 1
                keywords = self.nlp(keywords) 
                for i in keywords:
                    cnt[i] +=1
        from itertools import dropwhile

        for key, count in dropwhile(lambda key_count: key_count[1] > 5, cnt.most_common()):
            del cnt[key]
        del cnt['alzheimers disease']
        del cnt['dementia']
        print(cnt)

        # for key, count in dropwhile(lambda key_count: key_count[1] <  1500, cnt.most_common()):
        #     del cnt[key]
        #print(json.dumps(dict(cnt), indent=4))
        # print(summ/num)
        # cnt2 = Counter()
        # for i in range(4, 4797):
        #     for j in cnt.values():
        #         if i == j:
        #             cnt2[i] += 1
        # print(json.dumps(dict(cnt2), indent=4))
                
        
                

    def custom_categorize(self):
        df = pd.DataFrame(columns=['paper_obj', 'abstract', 'n_keywords'])
        with db_session:
            paper_list = select(p for p in Papers if p.abstract != None)[:]

        overall_list = []
        for paper in paper_list:
            abst_words = self.nlp(paper.abstract.split(' '))
            custom_keywords_raw = self.ngram_analyze(abst_words)
            custom_keywords_pro = []
            for c in custom_keywords_raw:
                custom_keywords_pro += c
            # lst = [paper, paper.abstract, ]

    def get_valid_set(self):
        paper_list_id = []
        paper_list = []
        with db_session:
            for idx in self.min5_ids:
                if Papers.get(id=np.asscalar(idx)):
                    if Papers.get(id=np.asscalar(idx)).degree == 0:
                        paper_list.append(Papers.get(id=np.asscalar(idx)))
        return paper_list

    def howmany(self):
        count = 0
        for paper in self.data:
            if paper.keywords != None:
                count+=1 
        print(count)
x = PaperAnalysis()
#x.howmany()
x.categorize()
#print((x.data[0].keywords))

# 204816 total amount of papers
# 106556 papers with keywords
# 198406 papers with abstracts
# 105595 papers with both keywords and abstracts
# 124413 unique keywords

# Using Clustered Info
# 16118 amount of papers with degree = 0 and belong to clustered info csv

