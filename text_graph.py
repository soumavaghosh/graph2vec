import re
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter, defaultdict
from graph_struct import Graph

stop_words = list(set(stopwords.words('english')))
rem = [',', '.', '?', ':', '...', '-', '"', "'", '!', "'s", "'nt", "'m", "'ve", "'d", '``', '\'\'', "'re", '(', ')', 'n\'t']
stop_words.extend(rem)

with open('../amazonreviews/test.ft.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()[:100]

label = ['0' if x.startswith('__label__1 ') else '1' for x in data]
data = [x[11:] for x in data]

with open('../amazonreviews/amazon_graph_indicator.txt', 'w') as f:
    f.write('\n'.join(label))

for i in range(len(data)):
    if 'www.' in data[i] or 'http:' in data[i] or 'https:' in data[i] or '.com' in data[i]:
        data[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", data[i])
    data[i] = sent_tokenize(data[i])
    data[i] = [word_tokenize(x.lower()) for x in data[i]]
    for j in range(len(data[i])):
        data[i][j] = [x for x in data[i][j] if x not in stop_words]

vocab = []
for d in data:
    for s in d:
        vocab.extend(s)

vocab = dict(Counter(vocab))

word_to_id = {list(vocab.keys())[i]:i+1 for i in range(len(vocab))}
id_to_word = {i+1:list(vocab.keys())[i] for i in range(len(vocab))}

graph_id = {}
window = 4
ct = 1

for d in data:
    g = defaultdict(list)
    sub_v = []
    for s in d:
        sub_v.extend(s)
        for i in range(len(s)-1):
            seq = s[i+1:min(i+1+window, len(s))]
            seq = [word_to_id[x] for x in seq]
            if word_to_id[s[i]] in list(g.keys()):
                g[word_to_id[s[i]]].extend(seq)
            else:
                g[word_to_id[s[i]]] = seq
    gph = Graph(len(list(set(sub_v))))

    for i in list(g.keys()):
        g[i] = sorted(list(set(g[i])))

    gph.graph = g
    graph_id[ct] = gph
    ct+=1
    print('done')


print('done')