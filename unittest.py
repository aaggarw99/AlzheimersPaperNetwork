from paper_analysis import PaperAnalysis
from cluster_analysis import ClusterAnalysis
from analysis_tools import AnalysisTools

import unittest

class PaperAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.pa = PaperAnalysis()

    def test_get_paper_from_id(self):
        paper = self.pa.get_paper_from_id(15325)
        self.assertEqual(paper[1], 'Expression cloning of LDLB, a gene essential for normal Golgi function and assembly of the ldlCp complex')
        self.assertEqual(paper[3], None)
        self.assertEqual(paper[4], 1999)
        self.assertEqual(paper[5], 2)
        
    def test_get_keywords(self):
        self.assertEqual(self.pa.get_keywords(-5), [])
        self.assertEqual(self.pa.get_keywords(97526), [])
        self.assertEqual(len(self.pa.get_keywords(24939)), 3)

    def test_get_title(self):
        self.assertEqual(len(self.pa.get_title(15209)),9)
        self.assertEqual(self.pa.get_title(-5), [])

    def test_get_abstract(self):
        self.assertEqual(self.pa.get_abstract(-5), [])

class ClusterAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.ca =  ClusterAnalysis("pubmedID_min5clusters_v2.csv")
        self.at = AnalysisTools()

    def test_find_keywords(self):
        all_keywords = self.ca.find_keywords(-1)
        all_comm = self.at.most_common(all_keywords, 3)
        self.assertEqual(all_comm[0][0], 'alzheimers disease')
        # self.at.show_wordcloud(all_keywords)

        keywords_66_list = self.ca.find_keywords(66)
        print(self.at.most_common(keywords_66_list, 3))
        # self.at.show_wordcloud(keywords_66_list, "66") 

    def test_find_titles(self):
        all_t_w_p = self.ca.find_titles(-1)
        # self.at.show_wordcloud(all_t_w_p, feature="words in titles")
        all_t_w_p_66 = self.na.find_titles(66)
        # self.na.show_wordcloud(all_t_w_p_66, feature="words in titles", subset="66")

    def test_find_abstract(self):
            abstract_all = self.ca.find_abstract(-1)
            self.ca.ngram_analyze(abstract_all)

            abstract_3390 = self.ca.find_abstract(3395)
            self.at.ngram_analyze(abstract_3390, model="student_t")
        
if __name__ == '__main__':
    unittest.main()


