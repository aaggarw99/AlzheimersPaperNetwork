# AlzheimersPaperNetwork
Using a subset of pubmed (<https://www.ncbi.nlm.nih.gov/pubmed/>) publications related to Alzheimer's, this network, clustered by similarities in citations using a SpeakEasy algorithm, is intended to find similarities between unpopular and popular papers in the alzheimer's field to potentially find unpopular papers that actually have significant importance.
After clustering, this project provides functions to analyze each cluster individually and the corresponding papers.

### By looking at only the titles, abstracts, and keywords of a paper (to reduce scan time), the following analyses and processes are conducted:
  * wordclouds visualizing the words found in individual clusters
  * finding most import words in the abstracts of publications using trigram analysis with student_t, jacard, chi_sq, and pmi distributions
  * natural language processing

#### NOTE: the actual compiled and filtered databases with the PubMed publications has not been uploaded.
