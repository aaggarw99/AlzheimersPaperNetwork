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


class PaperAnalysis():

    def __init__(self):
        pass
    
    def get_paper_from_id(self, idx, with_a_k=False):
        """
        Args:
        -----
        idx = id of paper
        with_a_k = when set to true, will only return paper information if it has an abstract and keywords

        Return:
        --------
        List of paper attributes in the following format:
        [id, title, abstract, keywords, year, month]

        """
        with db_session:
            if with_a_k==False:
                if (select(p for p in Papers if p.id == idx)[:]):
                    paper = select((p.id, p.title, p.abstract, p.keywords, p.year, p.month) for p in Papers if p.id == idx)[:]
                    return paper[0]
                else:
                    print("Paper does not exist in the database!")
                    return
            else:
                if (select(p for p in Papers if p.id == idx and p.abstract!=None and p.keywords!=None)[:]):
                    paper = select((p.id, p.title, p.abstract, p.keywords, p.year, p.month) for p in Papers if p.id == idx)[:]
                    return paper[0]
                else:
                    print("The paper requested has either no abstract or keywords")
                    return

                
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

        # look at the entire database keywords
        if idx == -1:
            with db_session:
                for k in (select(p.keywords for p in Papers if p.keywords != None)):
                    all_keywords += k.split(',')

                return all_keywords
        
        with db_session:
            # check to see if paper id exists in database
            if (select(p for p in Papers if p.id == idx)[:]):
                for k in (select(p.keywords for p in Papers if p.id == idx and p.keywords != None)[:]):
                    all_keywords += k.split(",")
                return all_keywords
            else:
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

        if idx == -1:
            with db_session:
                all_titles = select(p.title for p in Papers if p.title != None or p.title!="")[:]
                for t in all_titles:
                    all_title_words += t.split(' ')

                return all_title_words
            
        with db_session:
            # check to see if paper id exists in database
            if (select(p for p in Papers if p.id == idx)[:]):
                title = select(p.title for p in Papers if p.id == idx and p.title != None and p.title != "")[:]
                # if nothing was found, return an empty array. 
                if title:
                    return title[0].split()
                else:
                    return []
            else:
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

        if idx == -1:
            with db_session:
                all_abstract = select(p.abstract for p in Papers if p.abstract != None and p.abstract!="")[:]
                for a in all_abstract:
                    all_abstract_words += a.split(' ')
                return all_abstract_words
        else:    
            with db_session:
                # check to see if paper id exists in database
                if (select(p for p in Papers if p.id == idx)[:]):
                    # returns a list with one element (the  abstract string)
                    abstract = select(p.abstract for p in Papers if p.id == idx and p.abstract!=None and p.abstract!="")[:]
                    all_abstract_words = abstract[0].split(' ')

                    return all_abstract_words
                else:
                    return []

    def keyword_in_abstract(self, k_list, a_list):
        """
        Args:
        -----
        k_lst: list of keywords
        a_list: list of words in abstract

        Return:
        -------
        Percentage of how often keywords appear in the abstract
        """

        # removes duplicates in keywords list

        unique_k_list = list(set(k_list))

        total_keywords = len(unique_k_list)
        in_abstract = 0
        for i in range(total_keywords):
            if unique_k_list[i] in a_list:
                in_abstract += 1
                continue

        return (in_abstract / total_keywords)

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
        top = scores[:10]

        print(top)
    # LDA model
