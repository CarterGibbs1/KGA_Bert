import nltk
#import sys     # uncomment these lines if importing packages are weird. Also update paths
#sys.path.insert(0, r'c:\users\carte\appdata\local\programs\python\python310\lib\site-packages')
from wikidata.client import Client
from wikidata.entity import Entity
from collections import deque
import requests, json
import pandas as pd
import numpy as np
from tqdm import tqdm
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

MAX_LEVEL = 2

# UPDATE PATHS HERE
VEC_PATH = r"D:\SDS\KGA_Bert\data\KG_data\wikidata_translation_v1_vectors.npy\wiki_trans_v1_vec.npy"
LABEL_EMBEDDING_PATH = r'D:\SDS\KGA_Bert\data\KG_data\english_labels.tsv'
DATA_TRAIN_PATH = r'D:\SDS\KGA_Bert\data\glue_data\SST-2\train.tsv'
DATA_TEST_PATH = r'D:\SDS\KGA_Bert\data\glue_data\SST-2\dev.tsv'

def get_nouns(sentence):
    tokens = nltk.word_tokenize(sentence)
    tags = nltk.pos_tag(tokens)
    retVal = []
    i = 0
    while i < len(tags):
        key, tag = tags[i]
        if 'NN' in tag:
            retVal.append(key)
        i += 1
    return retVal

def get_wikidata_id(item):
    try:
        response = requests.get(f'https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&titles={item}&format=json')
        wikidata_id = list(json.loads(response.text)['query']['pages'].values())[0]['pageprops']['wikibase_item']
        return wikidata_id
    except:
        return None

embeddings_np = np.load(VEC_PATH)

def get_training_embeddings(labels_file):

    def __clean_str__(l : str):
        if '@en' in l:
            l = l.replace('@en', '')
        l = l.replace('"', '')
        return l.lower()
    
    retVal = {}
    with open(labels_file) as labels:
        for line in labels:
            curr_label, line_num = line.split('\t')
            line_num = int(line_num)
            curr_label = __clean_str__(curr_label)
            retVal[curr_label] = embeddings_np[line_num]
    return retVal

label_to_embedding = get_training_embeddings(LABEL_EMBEDDING_PATH)

def contains(noun):
    return noun.title() in label_to_embedding or noun.lower() in label_to_embedding or noun.upper() in label_to_embedding

def get_val(ent):
    val = label_to_embedding.get(str(ent.label))
    if val is None:
        val = label_to_embedding.get(str(ent.label).lower())
    if val is None:
        val = label_to_embedding.get(str(ent.label).upper())
    return val

def bfs(noun):
    client = Client()
    id = get_wikidata_id(noun)
    e = client.get(id)
    seen = set()
    q = deque([(e, 0)])
    averages = {}
    num_elems = [0] * 3
    while q:
        ent, level = q.popleft()
        if ent in seen or level > MAX_LEVEL:
            continue
        seen.add(ent)
        try:
            if isinstance(ent, Entity) and contains(str(ent.label)):
                if level not in averages:
                    averages[level] = get_val(ent)
                else:
                    averages[level] += get_val(ent)
                num_elems[level] += 1

                e = list(client.get(ent.id).values())[:min(len(e), 20)]
                for ent in e:
                    q.append((ent, level + 1))
        except:
            continue
    return None if 0 not in averages else {level : np.vectorize(lambda vals : round(vals, 4))(averages[level] / num_elems[level]) for level in range(3)}

data_train = pd.read_csv(DATA_TRAIN_PATH, sep='\t', header=0)
data_test = pd.read_csv(DATA_TEST_PATH, sep='\t', header=0)

nouns = set()
for sentence in data_train['sentence']:
    for noun in get_nouns(sentence):
        nouns.add(noun)

for sentence in data_test['sentence']:
    for noun in get_nouns(sentence):
        nouns.add(noun)

nouns_with_embeddings = {word for word in nouns if word in label_to_embedding}

embeddings_dict = {}
for noun in tqdm(list(nouns_with_embeddings)):
    entities = bfs(noun)
    if not entities:
        continue
    embeddings_dict[noun.lower()] = [entities[0], entities[1], entities[2]]

pd.DataFrame.from_dict(embeddings_dict, orient='index', columns=['1', '2', '3']).to_csv()
