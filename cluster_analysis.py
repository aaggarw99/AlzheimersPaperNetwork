from paper_analysis import PaperAnalysis
from database import Words, PaperWordCount, Papers, Citations
from analysis_tools import AnalysisTools
from pony.orm import *
import pandas as pd
import pickle
import os

class ClusterAnalysis(PaperAnalysis):
    """
    Provides methods to analyze papers in clusters
    """
    def __init__(self, csv):
        super().__init__()
        if not os.path.exists("organized_dict.pickle"):
            self.df = pd.read_csv(csv, names=['PubID','ClusterNo'])
            self.parse_clustered_publications()

        self.organized_by_cluster = self.get_pickle("organized_dict.pickle")

    def parse_clustered_publications(self):
        """
        Stores and dumps a dictionary of all publications to their respective \
        clusters into a pickle
        """
        org_list = []
        # dict ClusterNo : {PubID-1, PubID-2, ... , PubID-n}
        org_dict = dict()
        for j in range(self.df["ClusterNo"].min(), self.df['ClusterNo'].max()):
            org_list.clear()
            org_list.append(self.df.loc[self.df['ClusterNo'] == j]['PubID'].values.tolist())
            org_dict[j] = org_list[0]
        pickle.dump(org_dict, open("organized_dict.pickle", "wb"))

    def get_pickle(self, filename):
        return pickle.load(open(filename, "rb"))

    # given a cluster number, can we find the keywords in the papers in the cluster
    def find_keywords(self, cluster_number):
        """
        Searches through dataset given either a cluster number or the entire set for \
        keywords assigned to each publication and returns a filtered set of keywords.
        
        Args:
        -----
        cluster_number : if looking for keywords in a certain cluster, input the cluster label
            a cluster_number of -1 will search entire database

        Returns:
        --------
        A list of all keywords (filtered). Returns an empty list if invalid cluster number.
        """
        all_words = []

        # if all keywords are desired
        if cluster_number == -1:
            all_words = PaperAnalysis.get_keywords(self, -1)
            return AnalysisTools.nlp(self, all_words)

        # checks to see if cluster number is valid
        if self.organized_by_cluster[cluster_number]:
            paper_list = self.organized_by_cluster[cluster_number]
            for paper_id in paper_list:
                # get keywords for each paper
                temp = PaperAnalysis.get_keywords(self, paper_id)
                all_words += temp

            # returns all words in a nlp'd list
            return AnalysisTools.nlp(self, all_words)

        else:
            print("Cluster number was not found!")
            return []

    def find_titles(self, cluster_number):
        """
        Searches through dataset given either a cluster number or the entire set for \
        words in titles and then processes/stems them.

        Args:
        -----
        cluster_number : if looking for keywords in a certain cluster, input the cluster label
            a cluster_number of -1 will search entire database

        Returns:
        --------
        Filtered list of words from a cluster's titles
        """
        all_titles = []
        all_title_words = []

        # if looking at all cluster information
        if cluster_number == -1:
            with db_session:
                all_title_words = PaperAnalysis.get_title(self, -1)

            # Natural Language Processing + Return
            return AnalysisTools.nlp(self, all_title_words)

        # if looking at a certain cluster
        paper_list = self.organized_by_cluster[cluster_number]
        for paper_id in paper_list:
            all_title_words += PaperAnalysis.get_title(self, paper_id)

        # Natural Language Processing + Return
        return AnalysisTools.nlp(self, all_title_words)


    def find_abstract(self, cluster_number):
        """
        Args:
        -----
        cluster_number : if looking for keywords in a certain cluster, input the cluster label
            a cluster_number of -1 will search entire database

        Return:
        -------
        A list of filtered words from clustered abstracts
        """

        all_abstract = []
        all_abstract_words = []

        if cluster_number == -1:
            print("test")
            all_abstract_words = PaperAnalysis.get_abstract(self, -1)
            # Natural Language Processing + Return
            return AnalysisTools.nlp(self, all_abstract_words)

        paper_list = self.organized_by_cluster[cluster_number]
        for paper_id in paper_list:
            all_abstract_words += PaperAnalysis.get_abstract(self, paper_id)

        return AnalysisTools.nlp(self, all_abstract_words)    


