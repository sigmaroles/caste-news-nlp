lower_caste_keywords = ['dalit', 'untouchable', 'sc_st', 'obc', 'lower_caste', 'minorities', 'backward_class']
upper_caste_keywords = ['upper_caste', 'brahmin', 'kshatriya', 'vaishya']

priviledge_keywords = ['government', 'democracy', 'election', 'college', 'education', 'scholar', 'merit', 'meritorious', 'employer', 'national', 'international', 'rich']
negative_aspect = ['poor', 'violence', 'disease', 'unhealthy', 'incident', 'crime']
positive_aspect = ['empathy', 'care', 'companion', 'friend', 'healthy', 'prosperity']

neutral_keywords = ['individual', 'man', 'woman', 'person', 'people', 'subject', 'object', 'human', 'market']

from gensim.models import Word2Vec
from gensim.models.phrases import Phraser
import numpy as np
import itertools
from collections import defaultdict, OrderedDict
import os
from scipy import spatial




def scan_processed_with_phraser_(path_journal_processed, journ, path_phraser_models):
    print ("Launching scan of "+journ)
    # load the phraser model of the given journal name from given path
    fpath = path_phraser_models + '/' + journ + '_00_bigramphraser'
    bigram = Phraser.load(fpath)
    
    # dict of dicts; filenames as key, dict of frequency {word:count} as value
    texts = {}
    
    # iterate over all data from that journal
    fnames = os.listdir(path_journal_processed)
    ln_fnames = len(fnames)
    with open('logfile.txt', 'w') as lfh:
        for i, fname in enumerate(fnames):
            text = []
            for line in open(os.path.join(path_journal_processed, fname)):
                text = text + bigram[line.split()]
            frequency = defaultdict(float)
            for word in text:
                frequency[word] += 1
            
            # add the frequency counts of this text file to master dict
            texts[fname] = frequency
            if i%150==0:
                lfh.write("Done scanning "+fname+" ; {} out of {}".format(i+1, ln_fnames) + '\n')
                lfh.flush()
    
    return texts



"""
sample of output:
 '2016-6-7_0h0m1s__1606951162.html.txt': 0.0,
 '2016-7-7_18h45m3s__1627334438.html.txt': 0.0,
 '2016-3-10_0h0m3s__75310207.html.txt': 0.0,
 '2017-3-3_18h55m4s__609229536.html.txt': 0.0,
 '2017-1-31_18h55m1s__-388621826.html.txt': 0.0,
 '2016-6-21_0h15m1s__-1174394998.html.txt': 1.9988387823104858,
 '2016-11-17_18h41m3s__-284676987.html.txt': 1.0,
 '2017-1-31_19h0m1s__1566972843.html.txt': 0.99805748462677,
 '2016-11-9_18h35m2s__-86299334.html.txt': 0.0,
 '2016-12-12_18h32m0s__660548271.html.txt': 2.9954553842544556,
 '2017-5-3_19h0m2s__1124014970.html.txt': 0.0,
 '2017-10-11_20h0m2s__2060593936.html.txt': 0.0,
 '2017-12-28_18h30m3s__-62115611.html.txt': 0.0,
"""
def count_words(texts, words_and_weights):
    
    # will be returned .. dict to hold "how much" the input cluster occured in any given date
    cluster_occurence_measure = {}
    
    for fname in texts.keys():
        count = 0.0
        for word in words_and_weights.keys() :
            if word in texts[fname].keys() :
                count = count + texts[fname][word] * words_and_weights[word]
        cluster_occurence_measure[fname] = count
        
    return cluster_occurence_measure



"""
returns nothing
adds stuff to word_dict; a dict of lists, each list contains tuples
not meant to be used externally
"""
def _word_add(word, wvmodel, word_dict, depth, topn=5):
    if depth==1:
        # wvm.most_similar returns a list of tuples e.g. ('life', 0.877867) ... this needs to be handled by calling function
        word_dict[word] = wvmodel.most_similar(word, topn=topn)
        return
    else:
        for wtuple in wvmodel.most_similar(word, topn=topn):
            aword, _ = wtuple
            _word_add(aword, wvmodel, word_dict, depth=depth-1, topn=topn)




"""
get top n words and their similarity measures from word2vec model related to keyword
basically, get_words generates a dict with words as keys, and similarities to seed word as values
e.g. get_words(wvm, 'life') generates

{u'believe': 0.9959285259246826,
 u'ever': 0.9852045774459839,
 u'feel': 0.9888585805892944,
 u'going': 0.9956641793251038,
 u'got': 0.9888046383857727,
 u'hard': 1.0,
 u'history': 0.9895085096359253,
 ....
"""
def get_words(wvm, keyword, depth=2, tn=5):
    vocab = wvm.vocab
    all_words = [x for x in vocab.keys()]
    words_dict = {}
    
    # recursively "discover" words related to keyword, and add them all to words_dict 
    if keyword in all_words:
        _word_add(keyword, wvm, words_dict, depth=depth, topn=tn)
    else:
        raise ValueError

    ret = {}
    for w in words_dict :
        # it's unclear why this 1.0 weight (?) is needed..but AF used it in every instance of word_add, so let's keep it
        ret[w] = 1.0
        # for all other words that were discovered...
        for ww in words_dict[w]:
            # ... add it to the result dict with weight
            ret[ww[0]]=ww[1]
    return ret


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

