lower_caste_keywords = ['dalit', 'untouchable', 'sc_st', 'obc', 'lower_caste', 'minorities', 'backward_class']
upper_caste_keywords = ['upper_caste', 'brahmin', 'kshatriya', 'vaishya']

priviledge_keywords = ['government', 'democracy', 'election', 'college', 'education', 'scholar', 'merit', 'meritorious', 'employer', 
'national', 'international', 'rich']
negative_aspect = ['poor', 'violence', 'disease', 'unhealthy', 'incident', 'crime']
positive_aspect = ['empathy', 'care', 'companion', 'friend', 'healthy', 'prosperity']

neutral_keywords = ['individual', 'man', 'woman', 'person', 'people', 'subject', 'object', 'human', 'market']

import numpy as np
from scipy import spatial

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

def recurse_add_dict(word, wvmodel, worddict, depth=1, topn=5):
    alist = wvmodel.most_similar(word, topn=topn)
    worddict[word] = alist
    if depth==1:
        return
    else:
        for aword in alist:
            recurse_add_dict(aword, wvmodel, worddict, depth=depth-1, topn=topn)


def get_words(wvm, keywords, depth, tn):
    vocab = wvm.vocab
    all_words = [x for x in vocab.keys()]
    l_words_dict = {}
    for word in keywords:
        if word in all_words:
            recurse_add_dict(word, wvm, l_words_dict, depth=depth, topn=tn)
        else:
            print ("Word "+word+" not found.")

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
    t2 = wvm[list(list2.keys())[0]]
    l2 = len(l2_keys)
    for i in range(1,l2):
        t2 = np.sum((t2,wvm[l2_keys[i]]),axis=0)
    return 1 - spatial.distance.cosine(t1/l1, t2/l2)


if __name__=='__main__':
    print ("hola")


"""
Our question/conclusion:
How closely are the set of negative and positive words associated with
* lower caste?
* neutral topics?
* upper caste?
"""