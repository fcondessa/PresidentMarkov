import json
import networkx as nx
from collections import defaultdict
from random import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def create_markov_network_2(test_data = data,start_word=start_word,stop_word=stop_word):
    markov_network = defaultdict(lambda: defaultdict(int))
    set_words = [start_word,stop_word]
    for datum in test_data:
        try:
            list_words = datum['text'].split()
            list_words = [elem.lower() for elem in list_words]
            for elem in list_words:
                set_words.append(elem)
            rax = list_words
            aux = [((start_word,start_word),rax[0]),((start_word,rax[0]),rax[1])]
            aux += [((rax[i], rax[i + 1]), rax[i + 2]) for i in range(len(rax) - 2)]
            aux += [((rax[-2],rax[-1]),stop_word),((rax[-1],stop_word),stop_word)]
            for trio in aux:
                markov_network[trio[0]][trio[1]] +=1
        except IndexError:
            pass
    return markov_network,list(set_words)

def create_markov_network_3(test_data = data,start_word=start_word,stop_word=stop_word):
    markov_network = defaultdict(lambda: defaultdict(int))
    set_words = [start_word,stop_word]
    for datum in test_data:
        try:
            list_words = datum['text'].split()
            list_words = [elem.lower() for elem in list_words]
            for elem in list_words:
                set_words.append(elem)
            rax = list_words
            aux = [((start_word,start_word,start_word),rax[0]),((start_word,start_word,rax[0]),rax[1]),((start_word,rax[0],rax[1]),rax[2])]
            aux += [((rax[i], rax[i + 1],rax[i+2]), rax[i + 3]) for i in range(len(rax) - 3)]
            aux += [((rax[-3],rax[-2],rax[-1]),stop_word),((rax[-2],rax[-1],stop_word),stop_word)]
            for trio in aux:
                markov_network[trio[0]][trio[1]] +=1
        except IndexError:
            pass
    return markov_network,list(set_words)




def compute_probability_tweet(tweet,mkv_nw,start_word=start_word,stop_word=stop_word):
    words = tweet.lower().split()
    #prev_word = start_word
    prev_word = words[0]
    prob = 0
    for word in words[1:]:
        cur_word = word
        val = np.log((1.0*mkv_nw[prev_word][cur_word]) / (1.0*sum(mkv_nw[prev_word].values())))
        prev_word = word
        prob += val
    return prob

def randomly_create_tweet_2(markov_network,start_word=start_word,stop_word=stop_word,val_pow=1.0):
    tweet = []
    word = start_word
    prev_word = start_word
    while word != stop_word:
        list_possible_words = markov_network[(prev_word,word)].keys()
        val = markov_network[(prev_word,word)].values()
        val = [elem**val_pow for elem in val]
        val = np.cumsum(val)/(1.0*np.sum(val))
        g = random()
        idx = np.where(val >= g)[0][0]
        prev_word = word
        word = list_possible_words[idx]
        tweet.append(word)
    return ' '.join(tweet[:-1]).replace('&amp;','&')

def randomly_create_tweet_3(markov_network,start_word=start_word,stop_word=stop_word,val_pow=1.0):
    tweet = []
    word = start_word
    prev_word = start_word
    prev_prev_word = start_word
    while word != stop_word:
        list_possible_words = markov_network[(prev_prev_word,prev_word,word)].keys()
        val = markov_network[(prev_prev_word,prev_word,word)].values()
        val = [elem**val_pow for elem in val]
        val = np.cumsum(val)/(1.0*np.sum(val))
        g = random()
        idx = np.where(val >= g)[0][0]
        prev_prev_word = prev_word
        prev_word = word
        word = list_possible_words[idx]
        tweet.append(word)
    return ' '.join(tweet[:-1]).replace('&amp;','&')

# create list of datas according to years
markov_network,list_words = create_markov_network(data)
lol = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]

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
# G_stat_1 = [np.mean(nx.degree(elem).values()) for elem in G_list]
# G_stat_2 = [nx.average_clustering(nx.Graph(elem)) for elem in G_list]
# G_stat_3 = [nx.number_strongly_connected_components(elem) for elem in G_list]
# G_stat_4 = [nx.number_attracting_components(elem) for elem in G_list]
#

# for iter_i in range(len(networks)):
#     v = randomly_create_tweet(networks[iter_i], val_pow=2)
#     #print data_groups[iter_i][-1]['created_at'][4:10] + '->' + data_groups[iter_i][0]['created_at'][4:10]
#     print data_groups[iter_i][-1]['created_at'][4:8] + data_groups[iter_i][-1]['created_at'][-4:] + ' -> ' + data_groups[iter_i][0]['created_at'][4:8] + data_groups[iter_i][0]['created_at'][-4:]
#     print v

#complete_mkv_nw, list_wds = create_markov_network(data)


#
# prob_tweet = [compute_probability_tweet(tweet['text'],complete_mkv_nw) for tweet in data]
# times = [pd.to_datetime(tweet['created_at']) for tweet in data]
# mint = min(times)
# timesx = [(elem - mint).days for elem in times]
# z = np.polyfit(timesx,prob_tweet,10)
# p = np.poly1d(z)
# plt.plot(timesx,prob_tweet)
# plt.plot(timesx,p(timesx),'r--')

# test two hop network



# v = randomly_create_tweet(markov_network)
# print v

# test_data = data[::100]
# datum = test_data[0]
# rax = datum['text'].lower().split()

#(duo_nw,list_words) = create_markov_network_2(data)

#v = randomly_create_tweet_2(duo_nw,val_pow=2);

(trio_nw,list_words) = create_markov_network_3(data)
for i in range(2000):
    v = randomly_create_tweet_3(trio_nw,val_pow=1);
    print v


(duo_nw,list_words) = create_markov_network_2(data)
for i in range(200):
    v = randomly_create_tweet_2(duo_nw,val_pow=1);
    print v

# out_file_path = 'tweets.txt'
# with open(out_file_path,'wb') as f:
#     for iter_i in range(1000000):
#         if iter_i% 1000 ==0:
#             print iter_i
#         v = randomly_create_tweet_2(duo_nw,val_pow=2);
#         f.write('-'+v.encode('utf8')+'\n')
