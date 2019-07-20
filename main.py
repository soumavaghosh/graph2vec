# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:04:22 2019

@author: c53130a
"""
from graph_struct import Graph

dataset = "NCI1"

with open('./'+dataset+'/'+dataset+'_A.txt', 'r') as f:
    edge = f.readlines()
    
with open('./'+dataset+'/'+dataset+'_graph_indicator.txt', 'r') as f:
    node = f.readlines()

with open('./'+dataset+'/'+dataset+'_node_labels.txt', 'r') as f:
    node_l = f.readlines()

node = [int(x.replace('\n','')) for x in node]
node_l = [x.replace('\n','') for x in node_l]
edge = [x.replace('\n','') for x in edge]

g = Graph(len(node))
for e in edge:
    e = e.split(',')
    g.addEdge(int(e[0].strip()), int(e[1].strip()))

g.graph = dict(sorted(g.graph.items()))

node_id_to_graph_id = {i+1 : node[i] for i in range(len(node))}
node_id_to_node_label = {i+1 : node_l[i] for i in range(len(node))}
node_id_to_graph_id_wl = {}
relabel_dict = {}

for i in range(len(node_id_to_graph_id)):
    visited = [False] * len(g.graph)
    s = g.getsub([i+1], 1, visited)
    l = sorted([node_id_to_node_label[x] for x in s])
    index = ','.join(l)
    index+='-' + node_id_to_node_label[i + 1]
    if not index in list(relabel_dict.keys()):
        relabel_dict[index] = len(relabel_dict)+1
        node_id_to_graph_id_wl[i+1] = len(relabel_dict)+1
    else:
        node_id_to_graph_id_wl[i+1] = relabel_dict[index]
    print(str(i+1)+' - '+str(len(relabel_dict)))

# print(relabel_dict)
