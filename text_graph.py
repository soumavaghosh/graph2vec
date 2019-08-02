import re
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter, defaultdict
from graph_struct_txt import Graph
import pickle

stop_words = list(set(stopwords.words('english')))
rem = [',', '.', '?', ':', '...', '-', '"', "'", '!', "'s", "'nt", "'m", "'ve", "'d", '``', '\'\'', "'re", '(', ')', 'n\'t']
stop_words.extend(rem)

with open('../amazonreviews/train.ft.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()

with open('../amazonreviews/test.ft.txt', 'r', encoding='utf-8') as f:
    data.extend(f.readlines())

data = data[:10000]

print('data read')

label = ['0' if x.startswith('__label__1 ') else '1' for x in data]
data = [x[11:] for x in data]

with open('../amazonreviews/amazon_graph_indicator.txt', 'w') as f:
    f.write('\n'.join(label))

print('started cleaning')
print(len(data))
for i in range(len(data)):
    if (i+1)%1000 == 0:
        print(i+1)
    if 'www.' in data[i] or 'http:' in data[i] or 'https:' in data[i] or '.com' in data[i]:
        data[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", data[i])
    data[i] = sent_tokenize(data[i])
    data[i] = [word_tokenize(x.lower()) for x in data[i]]
    for j in range(len(data[i])):
        data[i][j] = [x for x in data[i][j] if x not in stop_words]

print('data cleaned')

vocab = []
for i in range(len(data)):
    if (i+1)%1000 == 0:
        print(i+1)
    d = data[i]
    for s in d:
        vocab.extend(list(set(s)))

vocab = list(set(vocab))

word_to_id = {}
id_to_word = {}

print('creating map')

for i, wrd in enumerate(vocab):
    if (i + 1) % 1000 == 0:
        print(i + 1)
    word_to_id[wrd] = i
    id_to_word[i] = wrd

graph_id = {}
window = 4
ct = 1

print('creating graphs')

for d in data:
    g = defaultdict(list)
    sub_v = []
    for s in d:
        if len(s)==0:
            continue
        sub_v.extend(s)
        for i in range(len(s)-1):
            seq = s[i+1:min(i+1+window, len(s))]
            seq = [word_to_id[x] for x in seq]
            if word_to_id[s[i]] in list(g.keys()):
                g[word_to_id[s[i]]].extend(seq)
            else:
                g[word_to_id[s[i]]] = seq

        if word_to_id[s[-1]] not in list(g.keys()):
            g[word_to_id[s[-1]]] = []

    gph = Graph(len(list(set(sub_v))))

    for i in list(g.keys()):
        g[i] = sorted(list(set(g[i])))

    gph.graph = g
    graph_id[ct] = gph
    ct+=1

print('graphs created')

with open('amazon_graphs.json', 'wb') as f:
    pickle.dump(graph_id, f)

def get_encoding(g):

    enc = []
    enc_str = ''
    for i in list(g.keys()):
        c = sorted([x for x in g[i]])
        enc.append([i, c])

    enc = sorted(enc, key=lambda x: (x[0], -len(x[1])))

    for i in enc:
        enc_str += '##' + str(i[0])
        if len(i[1]) == 0:
            enc_str += '#_'
        else:
            enc_str += '#' + ','.join([str(x) for x in i[1]])
    return enc_str

graph_id_voc = {}

for g in list(graph_id.keys()):
    if (g + 1) % 1000 == 0:
        print(g + 1)
    lst = []
    for n in list(graph_id[g].graph.keys()):
        if len(graph_id[g].graph[n])==0:
            continue
        l = []
        graph_id[g].graph[n] = sorted(graph_id[g].graph[n])
        for d in range(1, 4):
            sub = graph_id[g].getsub(n,d,len(word_to_id))
            sub_enc = get_encoding(sub)
            l.append(sub_enc)
        l = list(set(l))
        lst.extend(l)
    graph_id_voc[g] = lst

print('graph encoding done')

with open('amazon_graph_voc_3.json', 'wb') as f:
    pickle.dump(graph_id_voc, f)

print('done')