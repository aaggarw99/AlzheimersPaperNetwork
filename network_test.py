import network_analyze as na

def test_find_keywords(all_):
    if all_ == True:
        # get all keywords
        all_keywords = na.find_keywords(all_=True)
        # get most common keyword in database
        print("3 Most common keywords in entire database:\n")
        na.most_common(all_keywords, 3)
    else:
        # display wordcloud of all keywords
        na.show_wordcloud(all_keywords)

        # get list of keywords for cluster 66
        keywords_66_list = na.find_keywords(66)

        # get most common keyword in cluster 66
        print("3 most common keywords in cluster 66:\n")
        na.most_common(keywords_66_list, 3)
        # display wordcloud of keywords in cluster 66
        na.show_wordcloud(keywords_66_list, "66")

def test_find_titles(all_):
    if all_ ==True:
        # For entire dataset
        # Get a list of words in titles (processed)
        all_t_w_p = na.find_titles(all_=True)
        # most common words in titles
        print("7 most common words in titles in entire database:\n")
        na.most_common(all_t_w_p, 7)
        # shows wordcloud
        na.show_wordcloud(all_t_w_p, feature="words in titles")
    else:
        # For Cluster 66
        # Get a list of words in titles in Cluster 66 (processed)
        all_t_w_p_66 = na.find_titles(66)
        # most common words in cluster 66 titles
        print("7 most common words in cluster 66 titles")
        na.most_common(all_t_w_p_66, 7)
        # shows wordcloud
        na.show_wordcloud(all_t_w_p_66, feature="words in titles", subset="66")

def test_find_abstract(all_):
    if all_ == True:
        # Gets list of filtered words from all abstracts in the database
        abstract_all = na.find_abstract(all_=True)
        # Uses student_t distribution and finds most important words in the database
        na.ngram_analyze(abstract_all)
    else:
        # Gets list of filtered words from cluster 3390
        abstract_3390 = na.find_abstract(3395)

        # Uses student_t distribution and finds most import words in cluster 66 abstracts
        na.ngram_analyze(abstract_3390, model="student_t")

def test_correlation_cluster(cluster_number):
    """
    Finds correlation between keywords and the words in the abstract of a cluster
    """
    cluster_no = cluster_number

    keywords_list = na.find_keywords(cluster_number=cluster_no)
    abstract_word_list = na.find_abstract(cluster_number=cluster_no)

    print(na.keyword_in_abstract(keywords_list, abstract_word_list))


def test_correlation_paper():
    paper = na.get_paper_from_id(15922, with_a_k=True)
    if paper:
        words = paper[3].split(",")
        abstract = paper[2].split()
        print(words, abstract)
        print(na.keyword_in_abstract(words, abstract))
