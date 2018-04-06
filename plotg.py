from gensim.models.wrappers import FastText
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib
import os

from util_kw import *
wordlist = lower_caste_keywords + priviledge_keywords

if __name__=='__main__':

    ftmodel_files = [x for x in filter(lambda x: x.endswith('.bin'), os.listdir('.'))]
    print ("Model files found : "+str(ftmodel_files))

    for modelname in ftmodel_files:
        wvm = FastText.load_fasttext_format(modelname)
        g1 = nx.Graph()
        for word in wordlist:
            recurse_add_(word, wvm, g1, depth=1, topn=3)

        print (str(len(g1.nodes())) + "nodes in graph")

        nx.draw(g1, with_labels = True)
        plt.title(modelname)
        plt.savefig('plot_'+modelname+'_.png')
        plt.show()
