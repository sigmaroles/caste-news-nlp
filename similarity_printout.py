

from gensim.models import Word2Vec
from util_kw import *
import numpy as np
import itertools
import os
from scipy import spatial


path_models = 'word2vec_models'
journalfilenames = ['IN-thehindu-opinion_with_phraser',
'IN-indianexpress-india_with_phraser',
 'IN-indianexpress-opinion_with_phraser',
 'IN-indianexpress-editorials_with_phraser',
 'IN-thetimesofindia_00_with_phraser',
 'IN-thetimesofindia_01_with_phraser']

if __name__=='__main__':
    print ("\n\n **** Upper caste keywords and negative aspect ****")
    print_similarities_between_k_lists(negative_aspect, upper_caste_keywords)
    
    print ("\n\n **** Lower caste keywords and negative aspect ****")
    print_similarities_between_k_lists(negative_aspect, lower_caste_keywords)

    print ("\n\n **** Lower caste keywords and priviledge keywords ****")
    print_similarities_between_k_lists(lower_caste_keywords, priviledge_keywords)

    print ("\n\n **** Neutral keywords and priviledge keywords ****")
    print_similarities_between_k_lists(neutral_keywords, )