from analysis_tools import AnalysisTools
from database import *

class PaperAnalysis(AnalysisTools):

    def __init__(self):
        super().__init__()
    
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

