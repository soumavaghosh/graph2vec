import hashlib
from graph_struct import Graph
from collections import Counter
import pickle

dataset = "MUTAG"

with open('./'+dataset+'/'+dataset+'_A.txt', 'r') as f:
    edge = f.readlines()
    
with open('./'+dataset+'/'+dataset+'_graph_indicator.txt', 'r') as f:
    node = f.readlines()

with open('./'+dataset+'/'+dataset+'_node_labels.txt', 'r') as f:
    node_l = f.readlines()

node = [int(x.replace('\n', '')) for x in node]
node_l = [x.replace('\n', '') for x in node_l]
edge = [x.replace('\n', '') for x in edge]

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
    s = g.bfs([i+1], 1, visited)
    l = sorted([node_id_to_node_label[x] for x in s])
    index = ','.join(l)
    index += '-' + node_id_to_node_label[i + 1]
    index = int(hashlib.sha256(index.encode('utf-8')).hexdigest(), 16)%7
    if not index in list(relabel_dict.keys()):
        relabel_dict[index] = len(relabel_dict)+1
        node_id_to_graph_id_wl[i+1] = len(relabel_dict)+1
    else:
        node_id_to_graph_id_wl[i+1] = relabel_dict[index]
    print(str(i+1)+' - '+str(len(relabel_dict)))

def get_encoding(g):
    enc = []
    enc_str = ''
    for i in list(g.keys()):
        c = sorted([node_id_to_graph_id_wl[x] for x in g[i]])
        enc.append([node_id_to_graph_id_wl[i], c])

    enc = sorted(enc, key=lambda x: (x[0], -len(x[1])))

    for i in enc:
        enc_str += '##' + str(i[0])
        if len(i[1]) == 0:
            enc_str += '#_'
        else:
            enc_str += '#' + ','.join([str(x) for x in i[1]])
    return enc_str


graph_enc = {}
graph_voc = []
graph_id = dict(Counter(node))
graph_node_list = {}
st = 1
for i in sorted(graph_id):
    graph_node_list[i] = list(range(st, st+graph_id[i]))
    st = graph_node_list[i][-1]+1

for i in list(graph_node_list.keys()):
    lst = []
    for j in graph_node_list[i]:
        print(str(i)+' - '+str(j))
        for d in range(1, 4):
            sub = g.getsub(j, d)
            sub_enc = get_encoding(sub)
            lst.append(sub_enc)
    graph_enc[i] = lst
    graph_voc.extend(lst)

graph_voc = dict(Counter(graph_voc))

with open('./'+dataset+'/graph_voc_3.json', 'wb') as f:
    pickle.dump(graph_enc, f)

print('Completed')
