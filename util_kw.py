lower_caste_keywords = ['dalit', 'untouchable', 'sc_st', 'obc', 'lower_caste', 'minorities', 'backward_class']
upper_caste_keywords = ['upper_caste', 'brahmin', 'kshatriya', 'vaishya']

priviledge_keywords = ['government', 'democracy', 'election', 'college', 'education', 'scholar', 'merit', 'meritorious', 'employer', 'national', 'international', 'rich']
negative_aspect = ['poor', 'violence', 'disease', 'unhealthy', 'incident', 'crime']
positive_aspect = ['empathy', 'care', 'companion', 'friend', 'healthy', 'prosperity']

neutral_keywords = ['individual', 'man', 'woman', 'person', 'people', 'subject', 'object', 'human', 'market']

from gensim.models import Word2Vec
from util_kw import *
import numpy as np
import itertools
import os
from scipy import spatial

# blah update .. to test git script another

def add_to_graph(word, assoclist, graph):
    graph.add_node(word)
    for wtuple in assoclist:
        kword, kweight = wtuple
        graph.add_node(kword)
        graph.add_edge(word, kword, weight=kweight)
        
def recurse_add_(word, wvmodel, graph, depth=1, topn=5):
    if depth==1:
        # call add_to_graph and return
        alist = wvmodel.most_similar(word, topn=topn)
        add_to_graph(word, alist, graph)
        return
    else:
        # generate wordlist, then call recurse_add_ with each word in wordlist, with depth-1
        alist = wvmodel.most_similar(word, topn=topn)
        for wtuple in alist:
            aword, _ = wtuple
            recurse_add_(aword, wvmodel, graph, depth=depth-1, topn=topn)


def word_add_(word, wvmodel, lista, depth=1, topn=5):
    if depth==1:
        lista[word] = wvmodel.most_similar(word, topn=topn)
        return
    else:
        for wtuple in wvmodel.most_similar(word, topn=topn):
            aword, _ = wtuple
            word_add_(aword, wvmodel, lista, depth=depth-1, topn=topn)


"""
get all words from word2vec model(wvm) within topn = tn related to the word keyword
"""
def get_words(wvm, keyword, depth=2, tn=5):
    vocab = wvm.vocab
    all_words = [x for x in vocab.keys()]
    l_words_dict = {}
    
    if keyword in all_words:
        word_add_(keyword, wvm, l_words_dict, depth=depth, topn=tn)
    else:
        #print ("Word "+keyword+" not found.")
        raise ValueError

    l_words_simpl = {}
    for w in l_words_dict :
        l_words_simpl[w] = 1.0
        for ww in l_words_dict[w]:
            l_words_simpl[ww[0]]=ww[1]
    return l_words_simpl


def similarity(list1,list2, wvm):
    l1_keys = list(list1.keys())
    l2_keys = list(list2.keys())
    
    t1 = wvm[l1_keys[0]]
    l1 = len(l1_keys)
    for i in range(1,l1):
        t1 = np.sum((t1,wvm[l1_keys[i]]),axis=0)
    t2 = wvm[l2_keys[0]]
    l2 = len(l2_keys)
    for i in range(1,l2):
        t2 = np.sum((t2,wvm[l2_keys[i]]),axis=0)
    return 1 - spatial.distance.cosine(t1/l1, t2/l2)

path_models = 'word2vec_models'
journalfilenames = ['IN-thehindu-opinion',
    'IN-indianexpress-india',
    'IN-indianexpress-opinion',
    'IN-indianexpress-editorials',
    'IN-thetimesofindia_00',
    'IN-thetimesofindia_01']

def print_similarities_between_k_lists(klist1, klist2):
    for journal in journalfilenames:
        wvm = Word2Vec.load(path_models+'/'+journalfilenames[-1]+'_with_phraser').wv
        permut1 = list(itertools.chain(itertools.product(klist1, klist2)))
        for pair in permut1:
            word1, word2 = pair
            try:
                list1 = get_words(wvm, word1)
                list2 = get_words(wvm, word2)
            except ValueError:
                continue
            sim = similarity(list1, list2, wvm)
            print ("Journal {0} ; between {1} and {2} is {3:.4f}".format(journal, word1, word2, sim))

if __name__=='__main__':
    
    print_similarities_between_k_lists(negative_aspect, upper_caste_keywords)

"""
Our question/conclusion:
How closely are the set of negative and positive words associated with
* lower caste?
* neutral topics?
* upper caste?
"""