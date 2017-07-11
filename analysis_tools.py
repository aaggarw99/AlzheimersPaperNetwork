from database import Words, PaperWordCount, Papers, Citations
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

class AnalysisTools():
    def __init__(self):
        pass

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

    from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder


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
