import json
import networkx as nx
from collections import defaultdict
from random import random
import numpy as np
import matplotlib.pyplot as plt

data_path = 'data/input_data.json'

class MC:
    def __init__(self):
        self.data_path = data_path

def load_data(data_path):
    with open(data_path) as f:
        return json.load(f)

data = load_data(data_path)

start_word = 'TWEET_START'
stop_word = 'TWEET_STOP'


#test_data = data[:2000]

def create_markov_network(test_data = data,start_word=start_word,stop_word=stop_word):
    markov_network = defaultdict(lambda: defaultdict(int))
    set_words = [start_word,stop_word]
    for datum in test_data:
        list_words = datum['text'].split()
        list_words = [elem.lower() for elem in list_words]
        for elem in list_words:
            set_words.append(elem)
        markov_network[start_word][list_words[0]]+=1
        for iter_i in range(len(list_words)-1):
            markov_network[list_words[iter_i]][list_words[iter_i + 1]] +=1
        markov_network[list_words[-1]][stop_word] += 1
    return markov_network,list(set_words)

markov_network,list_words = create_markov_network(data)

def randomly_create_tweet(markov_network,start_word=start_word,stop_word=stop_word,val_pow=1.0):
    tweet = []
    word = start_word
    while word != stop_word:
        list_possible_words = markov_network[word].keys()
        val = markov_network[word].values()
        val = [elem**val_pow for elem in val]
        val = np.cumsum(val)/(1.0*np.sum(val))
        g = random()
        idx = np.where(val >= g)[0][0]
        word = list_possible_words[idx]
        tweet.append(word)
    return ' '.join(tweet[:-1])


v = randomly_create_tweet(markov_network)
print v



# create list of datas according to years

lol = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]

def create_graph(markov_network,list_words):
    G = nx.DiGraph()
    G.add_nodes_from(list_words)
    for key1 in markov_network.keys():
        for key2 in markov_network[key1].keys():
            G.add_edge(key1,key2,weigth= markov_network[key1][key2])
    return G

def create_graph_cleaner(markov_network,list_words,start_word=start_word,stop_word=stop_word):
    G = nx.DiGraph()
    G.add_nodes_from(list_words)
    for key1 in markov_network.keys():
        for key2 in markov_network[key1].keys():
            G.add_edge(key1,key2,weigth= markov_network[key1][key2])
    G.remove_node(start_word)
    G.remove_node(stop_word)
    return G

G_list = []
data_groups = lol(data,1000)
networks=  []
for data_elem in data_groups:
    markov_network,list_words = create_markov_network(data_elem)
    networks.append(markov_network)
    #G = create_graph(markov_network, list_words)
    G = create_graph_cleaner(markov_network, list_words)
    G_list.append(G)

# v = randomly_create_tweet(markov_network, val_pow=10)
# print v


#
G_stat_1 = [np.mean(nx.degree(elem).values()) for elem in G_list]
G_stat_2 = [nx.average_clustering(nx.Graph(elem)) for elem in G_list]
G_stat_3 = [nx.number_strongly_connected_components(elem) for elem in G_list]
G_stat_4 = [nx.number_attracting_components(elem) for elem in G_list]


for iter_i in range(len(networks)):
    v = randomly_create_tweet(networks[iter_i], val_pow=2)
    #print data_groups[iter_i][-1]['created_at'][4:10] + '->' + data_groups[iter_i][0]['created_at'][4:10]
    print data_groups[iter_i][-1]['created_at'][4:8] + data_groups[iter_i][-1]['created_at'][-4:] + ' -> ' + data_groups[iter_i][0]['created_at'][4:8] + data_groups[iter_i][0]['created_at'][-4:]
    print v