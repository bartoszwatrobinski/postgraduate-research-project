from sentence_transformers import SentenceTransformer
from scipy import spatial
import numpy as np
import spacy
import logging
import torch
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from dataset import SignedBipartiteData
from scipy import spatial
from ast import literal_eval
from torch_scatter import scatter_add
from torch_geometric.utils.num_nodes import maybe_num_nodes
from sklearn.utils.class_weight import compute_class_weight
import os
from nltk.stem import WordNetLemmatizer
from scipy.stats import kendalltau
from scipy.spatial.distance import cosine

from collections import Counter

import json
from kneed import KneeLocator
import matplotlib.pyplot as plt

from tqdm import tqdm


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
lemmatizer = WordNetLemmatizer()

from collections import defaultdict

# Initialize a dictionary to track the success and failure counts
embedding_stats = defaultdict(lambda: {'success': 0, 'failure': 0})

def get_bert_embedding(text, tokenizer, model, device):
    # the code generates fine-tuned BERT embedding 
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
    except Exception as e:
        logging.error(f"Failed to generate embedding for: {text}, Error: {e}")
        embedding_stats[text]['failure'] += 1
        
        return None  # Return None if the embedding fails

# the code used for filtering nouns and edges based on similarity to any of the subreddit titles
def filter_by_sim_bert(ents, sim, tokenizer, model, device):
    filtered = set()
    
    # Precompute embeddings for subreddit titles with a progress bar
    subreddit_titles = ['brexit', 'blm', 'climate', 'republican', 'democrat']
    title_embeddings = np.array([get_bert_embedding(t, tokenizer, model, device) for t in tqdm(subreddit_titles, desc="Generating Subreddit Title Embeddings")])

    # Iterate over each entity with a progress bar
    for e_orig in tqdm(ents, desc="Filtering Entities by Similarity"):
        cleaned_e = clean_text_for_bert(e_orig)
        bert_embed_e = get_bert_embedding(cleaned_e, tokenizer, model, device)
        
        if bert_embed_e is None:
            logging.warning(f"Failed to get embedding for entity: {e_orig}")
            embedding_stats[e_orig]['failure'] += 1
            continue

        embedding_success = False
        
        for bert_embed_t in title_embeddings:
            cos_sim = cosine_similarity(bert_embed_e, bert_embed_t.reshape(1, -1))[0][0]
            
            if cos_sim > sim:
                filtered.add(e_orig)
                embedding_success = True
                break  # Stop after the first match
            
        if embedding_success:
            embedding_stats[e_orig]['success'] += 1
        else:
            embedding_stats[e_orig]['failure'] += 1
    
    logging.info(f"Filtered entities count: {len(filtered)}")
    return list(filtered)
# the code used for filtering nouns and edges based on similarity to each of the subreddit titles
def filter_by_sim_bert_by_subreddit(ents, sim_thresholds, tokenizer, model, device):
    filtered = set()
    
    # Subreddit titles and corresponding similarity thresholds
    subreddit_titles = ['brexit', 'blm', 'climate', 'republican', 'democrat']
    title_embeddings = {}
    
    # Precompute embeddings for subreddit titles
    for title in subreddit_titles:
        title_embeddings[title] = get_bert_embedding(title, tokenizer, model, device)
    
    # Iterate over each entity
    for e_orig in tqdm(ents, desc="Filtering Entities by Similarity"):
        cleaned_e = clean_text_for_bert(e_orig)
        bert_embed_e = get_bert_embedding(cleaned_e, tokenizer, model, device)
        
        if bert_embed_e is None:
            logging.warning(f"Failed to get embedding for entity: {e_orig}")
            embedding_stats[e_orig]['failure'] += 1
            continue

        embedding_success = False

        
        for subreddit, bert_embed_t in title_embeddings.items():
            cos_sim = cosine_similarity(bert_embed_e, bert_embed_t.reshape(1, -1))[0][0]
            
            # Using subreddit-specific similarity threshold
            if cos_sim > sim_thresholds[subreddit]:
                filtered.add(e_orig)
                embedding_success = True
                break  

        if embedding_success:
            embedding_stats[e_orig]['success'] += 1
        else:
            embedding_stats[e_orig]['failure'] += 1
    
    logging.info(f"Filtered entities count: {len(filtered)}")
    return list(filtered)




def report_embedding_statistics(embedding_stats):
    total_words = sum(stats['success'] + stats['failure'] for stats in embedding_stats.values())
    total_failures = sum(stats['failure'] for stats in embedding_stats.values())

    failure_percentage = (total_failures / total_words) * 100 if total_words > 0 else 0

    logging.info(f"Total words processed: {total_words}")
    logging.info(f"Total failures: {total_failures}")
    logging.info(f"Percentage of words that failed to generate embeddings: {failure_percentage:.2f}%")


# function developed by Lorge et al. (2024) 
def get_cos_sim(sbert_model, entity, embed, embeddings_dict):
    # create pro/con sentences
    e_cap = entity.capitalize()
    pro = 'I am for ' + e_cap + '.'
    con = 'I am against ' + e_cap + '.'

    # encode pro/con sentences
    if pro in embeddings_dict:
      e_for = embeddings_dict[pro]
    else:
      e_for = sbert_model.encode(pro)
      embeddings_dict[pro] = e_for

    if con in embeddings_dict:
      e_con = embeddings_dict[con]
    else:
      e_con = sbert_model.encode(con)
      embeddings_dict[con] = e_con

    # get cosine sim with pro, cosine sim with con
    cos_for = 1 - spatial.distance.cosine(embed, e_for)
    cos_con = 1 - spatial.distance.cosine(embed, e_con)
    # get difference between pro and con cosine sims
    cos = cos_for - cos_con

    return cos
# function developed by Lorge et al. (2024) 
# function to create a graph of users and their pro/con cosine sim with entities in their posts
def get_pos_neg_edges(df):
    sbert_model = SentenceTransformer('all-mpnet-base-v2')
    embeddings_dict = {}
    nlp = spacy.load("en_core_web_md")
    excluded = ['CARDINAL', 'DATE', 'ORDINAL', 'WORK_OF_ART', 'PERCENT', 'QUANTITY', 'MONEY' ,'FAC', 'TIME', 'LANGUAGE',
              'PRODUCT']
    users_dict = {u: {} for u in set(df['author_parent'].unique()).union(set(df['author_child'].unique()))}
    u = len(users_dict)

    logging.info(f'Number of users: {u}')

    empty_parents = 0
    empty_children = 0

    # get posts
    for i in range(len(df)):
      print(i)
      # get entities
      parent = df['author_parent'].iloc[i]
      parent_doc = nlp(df['body_parent'].iloc[i])
      parent_ents = set([re.sub(r'[^\w\s]', '', e.text).lower() for e in parent_doc.ents if e.label_ not in excluded])

      child = df['author_child'].iloc[i]
      child_doc = nlp(df['body_child'].iloc[i])
      child_ents = set([re.sub(r'[^\w\s]', '', e.text).lower() for e in child_doc.ents if e.label_ not in excluded])

      # get embeddings
      if parent_ents != set():
        # parent_embed = np.mean(sbert_model.encode([s.text for s in list(parent_doc.sents)]), axis = 0)
        parent_embed = sbert_model.encode([s.text for s in list(parent_doc.sents)])
        if len(parent_embed) == 0:
          logging.error('Empty sentences parent')
      else:
        empty_parents+=1
      if child_ents != set():
        # child_embed = np.mean(sbert_model.encode([s.text for s in list(child_doc.sents)]), axis = 0)
        child_embed = sbert_model.encode([s.text for s in list(child_doc.sents)])
        if len(parent_embed) == 0:
          logging.error('Empty sentences child')
      else:
        empty_children+=1

      # parent entities cos sim
      for e in list(parent_ents):
        c = []
        for embed in parent_embed:
          cos_diff = get_cos_sim(sbert_model, e, embed, embeddings_dict)
          if not isinstance(cos_diff, np.floating):
            logging.error('cosine diff is not float')
          c.append(cos_diff)
        cos = np.nanmean(c)
        e = lemmatizer.lemmatize(e)
        if e in users_dict[parent]:
          users_dict[parent][e].append(cos)
        else:
          users_dict[parent][e] = [cos]

      # children entities cos sim 
      for e in list(child_ents):
        c = []
        for embed in child_embed:
          cos_diff = get_cos_sim(sbert_model, e, embed, embeddings_dict)
          if not isinstance(cos_diff, np.floating):
            logging.error('cosine diff is not float')
          c.append(cos_diff)
        cos = np.nanmean(c)
        e = lemmatizer.lemmatize(e)
        if e in users_dict[child]:
          users_dict[child][e].append(cos)
        else:
          users_dict[child][e] = [cos]

    pos_edges = {}
    neg_edges = {}
    all_cos= []

    # get mean cosine diff (it's heavily biased towards negative!)
    for u in users_dict:
      for e in users_dict[u]:
        all_cos.extend(users_dict[u][e])

    mean_cos_diff = np.nanmean(all_cos)
    logging.info(f'Mean cosine diff : {mean_cos_diff}')

    for u in users_dict:
      for e in users_dict[u]:
        # put average of cosine sims with entity in pos/neg edges dict
        mean_cos = np.nanmean(users_dict[u][e])
        if mean_cos > mean_cos_diff:
          if u in pos_edges:
            pos_edges[u][e] = mean_cos 
          else:
            pos_edges[u] = {e: mean_cos}
        # neg edges
        else:
          if u in neg_edges:
            neg_edges[u][e] = mean_cos
          else:
            neg_edges[u] =  {e: mean_cos}

    pos_l = sum(len(pos_edges[u]) for u in pos_edges) 
    neg_l = sum(len(neg_edges[u]) for u in neg_edges) 
    logging.info(f'Extracted {pos_l} positive edges and {neg_l} negative edges')

    return pos_edges, neg_edges 








# function for filtering NOUNS, adapted from get_pos_neg_edges
def get_pos_neg_edges_nouns(df):
    sbert_model = SentenceTransformer('all-mpnet-base-v2')
    embeddings_dict = {}
    nlp = spacy.load("en_core_web_md")
    lemmatizer = WordNetLemmatizer()
    
    users_dict = {u: {} for u in set(df['author_parent'].unique()).union(set(df['author_child'].unique()))}
    u = len(users_dict)
    logging.info(f'Number of users: {u}')
    empty_parents = 0
    empty_children = 0
    
    # Get posts
    for i in range(len(df)):
        print(i)
        # Extract relevant nouns directly from your list
        parent = df['author_parent'].iloc[i]
        parent_doc = nlp(df['body_parent'].iloc[i])
        parent_nouns = set([token.lemma_.lower() for token in parent_doc if token.lemma_.lower() in filtered_lemmatized_nouns.json])
        
        child = df['author_child'].iloc[i]
        child_doc = nlp(df['body_child'].iloc[i])
        child_nouns = set([token.lemma_.lower() for token in child_doc if token.lemma_.lower() in common_nouns])

        # Get embeddings
        if parent_nouns:
            parent_embed = sbert_model.encode([s.text for s in list(parent_doc.sents)])
            if len(parent_embed) == 0:
                logging.error('Empty sentences parent')
        else:
            empty_parents += 1
        
        if child_nouns:
            child_embed = sbert_model.encode([s.text for s in list(child_doc.sents)])
            if len(child_embed) == 0:
                logging.error('Empty sentences child')
        else:
            empty_children += 1
        
        # Parent nouns cosine similarity
        for noun in list(parent_nouns):
            c = []
            for embed in parent_embed:
                cos_diff = get_cos_sim(sbert_model, noun, embed, embeddings_dict)
                if not isinstance(cos_diff, np.floating):
                    logging.error('cosine diff is not float')
                c.append(cos_diff)
            cos = np.nanmean(c)
            noun = lemmatizer.lemmatize(noun)
            if noun in users_dict[parent]:
                users_dict[parent][noun].append(cos)
            else:
                users_dict[parent][noun] = [cos]
        
        # Children nouns cosine similarity 
        for noun in list(child_nouns):
            c = []
            for embed in child_embed:
                cos_diff = get_cos_sim(sbert_model, noun, embed, embeddings_dict)
                if not isinstance(cos_diff, np.floating):
                    logging.error('cosine diff is not float')
                c.append(cos_diff)
            cos = np.nanmean(c)
            noun = lemmatizer.lemmatize(noun)
            if noun in users_dict[child]:
                users_dict[child][noun].append(cos)
            else:
                users_dict[child][noun] = [cos]
    
    pos_edges = {}
    neg_edges = {}
    all_cos = []
    
    # Get mean cosine difference
    for u in users_dict:
        for noun in users_dict[u]:
            all_cos.extend(users_dict[u][noun])
    mean_cos_diff = np.nanmean(all_cos)
    logging.info(f'Mean cosine diff : {mean_cos_diff}')
    
    for u in users_dict:
        for noun in users_dict[u]:
            # Put average of cosine similarities with noun in pos/neg edges dictionary
            mean_cos = np.nanmean(users_dict[u][noun])
            if mean_cos > mean_cos_diff:
                if u in pos_edges:
                    pos_edges[u][noun] = mean_cos 
                else:
                    pos_edges[u] = {noun: mean_cos}
            else:
                if u in neg_edges:
                    neg_edges[u][noun] = mean_cos
                else:
                    neg_edges[u] = {noun: mean_cos}
    
    pos_l = sum(len(pos_edges[u]) for u in pos_edges) 
    neg_l = sum(len(neg_edges[u]) for u in neg_edges) 
    logging.info(f'Extracted {pos_l} positive edges and {neg_l} negative edges')
    
    return pos_edges, neg_edges

def get_pos_neg_edges_nouns(df):
    sbert_model = SentenceTransformer('all-mpnet-base-v2')
    embeddings_dict = {}
    nlp = spacy.load("en_core_web_md")
    lemmatizer = WordNetLemmatizer()
    
    users_dict = {u: {} for u in set(df['author_parent'].unique()).union(set(df['author_child'].unique()))}
    u = len(users_dict)
    logging.info(f'Number of users: {u}')
    empty_parents = 0
    empty_children = 0
    
    # Get posts
    for i in range(len(df)):
        print(i)
        # Extract relevant nouns directly from the text
        parent = df['author_parent'].iloc[i]
        parent_doc = nlp(df['body_parent'].iloc[i])
        parent_nouns = set([token.lemma_.lower() for token in parent_doc if token.pos_ == 'NOUN'])
        
        child = df['author_child'].iloc[i]
        child_doc = nlp(df['body_child'].iloc[i])
        child_nouns = set([token.lemma_.lower() for token in child_doc if token.pos_ == 'NOUN'])

        # Get embeddings
        if parent_nouns:
            parent_embed = sbert_model.encode([s.text for s in list(parent_doc.sents)])
            if len(parent_embed) == 0:
                logging.error('Empty sentences parent')
        else:
            empty_parents += 1
        
        if child_nouns:
            child_embed = sbert_model.encode([s.text for s in list(child_doc.sents)])
            if len(child_embed) == 0:
                logging.error('Empty sentences child')
        else:
            empty_children += 1
        
        # Parent nouns cosine similarity
        for noun in list(parent_nouns):
            c = []
            for embed in parent_embed:
                cos_diff = get_cos_sim(sbert_model, noun, embed, embeddings_dict)
                if not isinstance(cos_diff, np.floating):
                    logging.error('cosine diff is not float')
                c.append(cos_diff)
            cos = np.nanmean(c)
            noun = lemmatizer.lemmatize(noun)
            if noun in users_dict[parent]:
                users_dict[parent][noun].append(cos)
            else:
                users_dict[parent][noun] = [cos]
        
        # Children nouns cosine similarity 
        for noun in list(child_nouns):
            c = []
            for embed in child_embed:
                cos_diff = get_cos_sim(sbert_model, noun, embed, embeddings_dict)
                if not isinstance(cos_diff, np.floating):
                    logging.error('cosine diff is not float')
                c.append(cos_diff)
            cos = np.nanmean(c)
            noun = lemmatizer.lemmatize(noun)
            if noun in users_dict[child]:
                users_dict[child][noun].append(cos)
            else:
                users_dict[child][noun] = [cos]
    
    pos_edges = {}
    neg_edges = {}
    all_cos = []
    
    # Get mean cosine difference
    for u in users_dict:
        for noun in users_dict[u]:
            all_cos.extend(users_dict[u][noun])
    mean_cos_diff = np.nanmean(all_cos)
    logging.info(f'Mean cosine diff : {mean_cos_diff}')
    
    for u in users_dict:
        for noun in users_dict[u]:
            # Put average of cosine similarities with noun in pos/neg edges dictionary
            mean_cos = np.nanmean(users_dict[u][noun])
            if mean_cos > mean_cos_diff:
                if u in pos_edges:
                    pos_edges[u][noun] = mean_cos 
                else:
                    pos_edges[u] = {noun: mean_cos}
            else:
                if u in neg_edges:
                    neg_edges[u][noun] = mean_cos
                else:
                    neg_edges[u] = {noun: mean_cos}
    
    pos_l = sum(len(pos_edges[u]) for u in pos_edges) 
    neg_l = sum(len(neg_edges[u]) for u in neg_edges) 
    logging.info(f'Extracted {pos_l} positive edges and {neg_l} negative edges')
    
    return pos_edges, neg_edges


# function developed by Lorge et al. (2024) 
# function to create data list to feed to PyG dataloader using signed bipartite data class
# ! try to turn this into a generator and see if dataloader takes output?
def create_graph_data_lists(df, embeddings_dict, embed_size, pos_edges, neg_edges, edge_weights=True):
    logging.info(f'Using embeddings with size {embed_size}')
    pos_l = sum(len(pos_edges[u]) for u in pos_edges) 
    neg_l = sum(len(neg_edges[u]) for u in neg_edges)
    users = set(list(pos_edges.keys()) + list(neg_edges.keys()))
    ents = [list(pos_edges[u].keys()) for u in pos_edges] + [list(neg_edges[u].keys()) for u in neg_edges]
    entities = sorted(list(set([i for sub in ents for i in sub])))

    logging.info(f'Processing {len(users)} users')
    logging.info(f'Processing {len(entities)} total items')
    logging.info(f'Processing {pos_l} positive edges and {neg_l} negative edges')

    # Initialize lists for data
    parents_data_list = []
    children_data_list = []

    # Create a dictionary to store embeddings
    voc_dict = {}
    errors = []

    # Embed each entity/noun
    for idx, e in enumerate(entities):
        original_e = e
        if '/' in e:
            e = ' '.join(e.split('/'))
        else:
            e = re.sub(r'[^\w\s]', '', e).lstrip()  # Clean entity/noun text
        if e != '':
            try:
                if len(e.split()) == 1:
                    embed = embeddings_dict[e]
                else:
                    # Average embeddings for multi-word entities/nouns
                    words = [lemmatizer.lemmatize(re.sub(r'[^\w\s]', '', i.lower())) for i in e.split()]
                    mw = np.zeros((len(words), embed_size))
                    for idx, w in enumerate(words):
                        mw[idx, :] = embeddings_dict[w]
                    embed = np.mean(mw, axis=0)
                voc_dict[e] = embed
            except Exception:
                logging.error(f'Could not get embedding for: {original_e}, {e}')
                errors.append(e)
        else:
            logging.error(f'Empty string for: {original_e} {e}')
            errors.append(e)

    # Create neutral embedding for cases where an embedding is missing
    l = len(voc_dict)
    m = np.zeros((l, embed_size))
    for idx, e in enumerate(list(voc_dict.keys())):
        m[idx, :] = voc_dict[e]
    neutral_embed = np.mean(m, axis=0)
    voc_dict['NEUTRAL_ENTITY'] = neutral_embed
    logging.info(f'Neutral embedding shape: {neutral_embed.shape}')

    # Assign neutral embeddings to errors
    print('ERRORS:', errors)
    for e in errors:
        voc_dict[e] = neutral_embed

    # Process each row in the dataframe
    parent_ids = list(df.author_parent)
    children_ids = list(df.author_child)

    for idx in range(len(df)):
        parent_id = parent_ids[idx]
        child_id = children_ids[idx]

        # Initialize edge lists
        parent_pos_edges = [[], []]
        parent_pos_weight = []
        parent_neg_edges = [[], []]
        parent_neg_weight = []
        child_pos_edges = [[], []]
        child_pos_weight = []
        child_neg_edges = [[], []]
        child_neg_weight = []

        # Collect all entities/nouns for parent and child
        parent_entities = []
        child_entities = []

        for edge_list in [pos_edges, neg_edges]:
            if parent_id in edge_list:
                parent_entities.extend(sorted(list(edge_list[parent_id].keys())))
            if child_id in edge_list:
                child_entities.extend(sorted(list(edge_list[child_id].keys())))

        # Initialize user and entity/noun feature matrices
        parent_user_feat = torch.zeros((1, embed_size))
        child_user_feat = torch.zeros((1, embed_size))
        parent_entities_feat = torch.zeros((len(parent_entities), embed_size))
        child_entities_feat = torch.zeros((len(child_entities), embed_size))

        # Process parent entities/nouns
        for i, e in enumerate(parent_entities):
            if '/' in e:
                e = ' '.join(e.split('/'))
            else:
                e = re.sub(r'[^\w\s]', '', e).lstrip()
            parent_entities_feat[i, :] = torch.tensor(voc_dict[e])

            if parent_id in pos_edges and e in pos_edges[parent_id]:
                parent_pos_edges[0].append(i)
                parent_pos_edges[1].append(0)
                parent_pos_weight.append(pos_edges[parent_id][e])
            if parent_id in neg_edges and e in neg_edges[parent_id]:
                parent_neg_edges[0].append(i)
                parent_neg_edges[1].append(0)
                parent_neg_weight.append(neg_edges[parent_id][e])

        # Process child entities/nouns
        for i, e in enumerate(child_entities):
            if '/' in e:
                e = ' '.join(e.split('/'))
            else:
                e = re.sub(r'[^\w\s]', '', e).lstrip()
            child_entities_feat[i, :] = torch.tensor(voc_dict[e])

            if child_id in pos_edges and e in pos_edges[child_id]:
                child_pos_edges[0].append(i)
                child_pos_edges[1].append(0)
                child_pos_weight.append(pos_edges[child_id][e])
            if child_id in neg_edges and e in neg_edges[child_id]:
                child_neg_edges[0].append(i)
                child_neg_edges[1].append(0)
                child_neg_weight.append(neg_edges[child_id][e])

        # Create graph data structures for parents and children
        if edge_weights:
            parent_graph = SignedBipartiteData(x_s=parent_entities_feat, x_t=parent_user_feat,
                                               pos_edge_index=torch.tensor(parent_pos_edges, dtype=torch.int64),
                                               neg_edge_index=torch.tensor(parent_neg_edges, dtype=torch.int64),
                                               pos_edge_weight=torch.tensor(parent_pos_weight),
                                               neg_edge_weight=torch.tensor(parent_neg_weight))
            child_graph = SignedBipartiteData(x_s=child_entities_feat, x_t=child_user_feat,
                                              pos_edge_index=torch.tensor(child_pos_edges, dtype=torch.int64),
                                              neg_edge_index=torch.tensor(child_neg_edges, dtype=torch.int64),
                                              pos_edge_weight=torch.tensor(child_pos_weight),
                                              neg_edge_weight=torch.tensor(child_neg_weight))
        else:
            parent_graph = SignedBipartiteData(x_s=parent_entities_feat, x_t=parent_user_feat,
                                               pos_edge_index=torch.tensor(parent_pos_edges, dtype=torch.int64),
                                               neg_edge_index=torch.tensor(parent_neg_edges, dtype=torch.int64),
                                               pos_edge_weight=torch.ones((len(parent_pos_edges[0]))),
                                               neg_edge_weight=torch.ones((len(parent_neg_edges[0]))))
            child_graph = SignedBipartiteData(x_s=child_entities_feat, x_t=child_user_feat,
                                              pos_edge_index=torch.tensor(child_pos_edges, dtype=torch.int64),
                                              neg_edge_index=torch.tensor(child_neg_edges, dtype=torch.int64),
                                              pos_edge_weight=torch.ones((len(child_pos_edges[0]))),
                                              neg_edge_weight=torch.ones((len(child_neg_edges[0]))))

        parents_data_list.append(parent_graph)
        children_data_list.append(child_graph)

    return parents_data_list, children_data_list



def mean_pool(bert_model, ids, masks, segs):
    n_tokens_pad = (ids >= 106).float().sum(dim=-1)
    output_bert = bert_model(ids, attention_mask=masks, token_type_ids=segs)[0]
    output_bert_pad = output_bert * (ids >= 106).float().unsqueeze(-1).expand(-1, -1, 768)
    output_bert_pooled = output_bert_pad.sum(dim=1) / n_tokens_pad.unsqueeze(-1).expand(-1, 768)
    return output_bert_pooled


def gcn_norm(edge_index, edge_weight=None, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    idx = col
    deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

# function developed by Lorge et al. (2024) 
def get_top_entities_by_freq(pos_edges, neg_edges, threshold):
    d = {}
    for u in pos_edges:
        for e in pos_edges[u]:
            if e in d:
                d[e] +=1
            else:
                d[e] = 1
    for u in neg_edges:
        for e in neg_edges[u]:
            if e in d:
                d[e] +=1
            else:
                d[e] = 1
    ents = [k for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)][:threshold]
    ents = [e for e in ents if len(e.split())==1]
    return ents

def get_all_entities(pos_edges, neg_edges):
    all_entities = set()  # Use a set to avoid duplicates

    # Add entities from positive edges
    for u in pos_edges:
        for e in pos_edges[u]:
            all_entities.add(e)

    # Add entities from negative edges
    for u in neg_edges:
        for e in neg_edges[u]:
            all_entities.add(e)

    # Convert the set to a sorted list (optional)
    all_entities = sorted(all_entities)

    return all_entities




def is_valid_noun(noun):
    # Exclude empty strings and further preprocessing 
    if not noun or noun in ["/s", "-", "", None]:
        return False
    # Excluding words with special characters
    if not re.match(r'^[a-zA-Z-]+$', noun):
        return False
    # If the noun passes all checks, it is considered valid
    return True

def get_top_nouns_by_freq(pos_edges, neg_edges, threshold):
    noun_counter = {}

    # Count nouns in positive edges
    for user in tqdm(pos_edges, desc="Counting Nouns in Positive Edges"):
        for noun in pos_edges[user]:
            if is_valid_noun(noun):
                if noun in noun_counter:
                    noun_counter[noun] += 1
                else:
                    noun_counter[noun] = 1

    # Count nouns in negative edges
    for user in tqdm(neg_edges, desc="Counting Nouns in Negative Edges"):
        for noun in neg_edges[user]:
            if is_valid_noun(noun):
                if noun in noun_counter:
                    noun_counter[noun] += 1
                else:
                    noun_counter[noun] = 1

    # Sort nouns by frequency and apply threshold
    top_nouns = [noun for noun, count in sorted(noun_counter.items(), key=lambda item: item[1], reverse=True)][:threshold]

    return top_nouns


#the code used for filtering out entities
def filter_ents(edges, ents):
    edges_filtered = {}
    for u in sorted(list(edges.keys())):
        new_u = {e: cos for e, cos in edges[u].items() if e in ents}
        if new_u != {}:
            edges_filtered[u] = new_u
    return edges_filtered

# function developed by Lorge et al. (2024) , used for developing mean edge weights and normalizing them by switching them from negative to positive
def normalise_edges(pos_edges, neg_edges):
    print('ORIG EDGES', len(pos_edges), len(neg_edges))
    neg_val = [neg_edges[u][e] for u in neg_edges for e in neg_edges[u]]
    pos_val = [pos_edges[u][e] for u in pos_edges for e in pos_edges[u]]
    all_val = neg_val + pos_val
    m = np.nanmean(all_val)
    print(f'mean of edge weights is {m}')
    neg_edges_filtered = {}
    pos_edges_filtered = {}
    
    # mean centering and swapping 'fake' negative edges back to pos edges 
    for u in sorted(list(neg_edges.keys())): # all values end up positive 
        new_u_neg = {e: np.abs(cos-m) for e, cos in neg_edges[u].items() if cos <= m} # real negatives
        new_u_pos = {e: np.abs(cos-m) for e, cos in neg_edges[u].items() if cos > m} # fake negatives 
        if new_u_neg != {}:
            neg_edges_filtered[u] = new_u_neg
        if u in pos_edges:
            old_u_pos = {e: np.abs(cos-m)  for e, cos in pos_edges[u].items()}
            new_u_pos.update(old_u_pos) # merging the two dict
            pos_edges_filtered[u] = new_u_pos
    for u in sorted(list(pos_edges.keys())):
        if u not in pos_edges_filtered:
            pos_edges_filtered[u] =  {e: np.abs(cos-m) for e, cos in pos_edges[u].items()}
    return pos_edges_filtered, neg_edges_filtered

# function developed by Lorge et al. (2024) 
def filter_by_sim(word2vec, ents, sim):
    filtered = set()
    for e_orig in ents:
        for t in ['brexit', 'blm', 'climate', 'republican', 'democrat']:
            try:
              if len(e_orig.split()) == 1:
                e = re.sub(r'[^\w]', '', e_orig)
                cos = 1 - spatial.distance.cosine(word2vec.wv[e], word2vec.wv[t])
              else:
                e = re.sub(r'[^\w\s]', ' ', e_orig)
                cos = np.mean([1 - spatial.distance.cosine(word2vec.wv[lemmatizer.lemmatize(w)], word2vec.wv[t]) for w in e.split()])
              if cos > sim:
                filtered.add(e)
            except Exception as exp:
              continue
    return list(filtered)

# function used for basic preprocessing before getting BERT embedding
def clean_text_for_bert(text):
    # Basic cleaning to remove unwanted characters
    text = re.sub(r'[^\w\s]', '', text)  
    text = text.strip()
    return text


# function developed by Lorge et. al (2024), however strongly edited with novel solutions: noun integration module and custom dataset filtering
def preprocess_edges_and_datasets(word2vec, df_train, df_valid, train_pos_edges, train_neg_edges,
                                  eval_pos_edges, eval_neg_edges,
                                  train_pos_edges_nouns, train_neg_edges_nouns,
                                  eval_pos_edges_nouns, eval_neg_edges_nouns,
                                  entity_threshold, noun_threshold,
                                  sim_thresholds_ents, sim_thresholds_nouns,
                                  lemmatise, filter_ents_dataset, filter_nouns_dataset, full_agreement, model, tokenizer, device, independent_filtering):


  logging.info(f'Initial df_train length: {len(df_train)}')
  logging.info(f'Initial df_valid length: {len(df_valid)}')
  
 # 1. Entities
  # a. Get top entities
  ents = get_top_entities_by_freq(train_pos_edges, train_neg_edges, threshold=entity_threshold)
  # b. Filter entities by similarity
  ents = filter_by_sim_bert_by_subreddit(ents, sim_thresholds=sim_thresholds_ents, tokenizer=tokenizer, model=model, device=device)

  print(f'Number of target entities: {len(ents)}')

  # c. Normalize entity edges
  train_pos_edges_norm, train_neg_edges_norm = normalise_edges(train_pos_edges, train_neg_edges)
  eval_pos_edges_norm, eval_neg_edges_norm = normalise_edges(eval_pos_edges, eval_neg_edges)

  # d. Filter normalized entity edges
  train_pos_edges_filtered = filter_ents(train_pos_edges_norm, ents) 
  train_neg_edges_filtered = filter_ents(train_neg_edges_norm, ents)
  eval_pos_edges_filtered = filter_ents(eval_pos_edges_norm, ents)
  eval_neg_edges_filtered = filter_ents(eval_neg_edges_norm, ents)

  # 2. Nouns
  # a. Load cached nouns
  nouns = get_top_nouns_by_freq(train_pos_edges_nouns, train_neg_edges_nouns, threshold=noun_threshold)
  # we eliminate all existing entities from the nouns list so that the lists become mutually exclusive
  #print("ENTS ", ents)
  #nouns = list(set(nouns) - set(ents))
    
  # b. Filter nouns based on threshold and similarity
  #nouns = [noun for noun, count in cached_nouns.items() if count >= noun_threshold]
  nouns = filter_by_sim_bert_by_subreddit(nouns, sim_thresholds=sim_thresholds_nouns, tokenizer=tokenizer, model=model, device=device)
  print(f'Number of target nouns: {len(nouns)}')
  print('TARGET NOUNS', sorted(nouns))

  # c. Compute edges for nouns
  #train_pos_edges_nouns, train_neg_edges_nouns = get_pos_neg_edges_nouns(df_train)
  #eval_pos_edges_nouns, eval_neg_edges_nouns = get_pos_neg_edges_nouns(df_valid)

  # d. Normalize noun edges
  train_pos_edges_nouns_norm, train_neg_edges_nouns_norm = normalise_edges(train_pos_edges_nouns, train_neg_edges_nouns)
  eval_pos_edges_nouns_norm, eval_neg_edges_nouns_norm = normalise_edges(eval_pos_edges_nouns, eval_neg_edges_nouns)

  # e. Filter normalized noun edges
  train_pos_edges_nouns_filtered = filter_ents(train_pos_edges_nouns_norm, nouns)
  train_neg_edges_nouns_filtered = filter_ents(train_neg_edges_nouns_norm, nouns)
  eval_pos_edges_nouns_filtered = filter_ents(eval_pos_edges_nouns_norm, nouns)
  eval_neg_edges_nouns_filtered = filter_ents(eval_neg_edges_nouns_norm, nouns)


      # 2.1 Check for overlapping edge names between entities and nouns
  entity_edge_names = set(train_pos_edges_filtered.keys()).union(train_neg_edges_filtered.keys())
  noun_edge_names = set(train_pos_edges_nouns_filtered.keys()).union(train_neg_edges_nouns_filtered.keys())
    
  overlapping_edges = entity_edge_names & noun_edge_names
  if overlapping_edges:
    logging.warning(f"Overlapping edge names found between entity and noun edges: {overlapping_edges}")
  else:
    logging.info("No overlapping edge names found between entity and noun edges.")

  # 3. Merge entity and noun edges
  train_pos_edges = merge_edges(train_pos_edges_filtered, train_pos_edges_nouns_filtered)
  train_neg_edges = merge_edges(train_neg_edges_filtered, train_neg_edges_nouns_filtered)
  eval_pos_edges = merge_edges(eval_pos_edges_filtered, eval_pos_edges_nouns_filtered)
  eval_neg_edges = merge_edges(eval_neg_edges_filtered, eval_neg_edges_nouns_filtered)

  combined_entities_nouns = set(ents + nouns)
  print(f'Total number of target items (entities + nouns): {len(combined_entities_nouns)}')
  print("TARGET ITEMS (ENTITIES + NOUNS)", combined_entities_nouns)
  with open('combined_entities_nouns.txt', 'w') as file:
        file.write('\n'.join(sorted(combined_entities_nouns)))

  print(f"Train positive edges: {len(train_pos_edges)}, Train negative edges: {len(train_neg_edges)}")
  print(f"Validation positive edges: {len(eval_pos_edges)}, Validation negative edges: {len(eval_neg_edges)}")



  # ents for dataset filtering by string
  ents_dataset = [f"'{e}'" for e in ents]
  print("Ents dataset: ", sorted(ents_dataset))

  nouns_dataset = [f"'{noun}'" for noun in nouns]
  print()
  print(sorted(nouns_dataset))

  combined_items_dataset = [f"'{item}'" for item in combined_entities_nouns]
  print(f"Combined items for dataset filtering: {sorted(combined_items_dataset)}")

  def to_set(x):
    if x!= 'set()':
      return literal_eval(x)
    else:
      return set()

  if lemmatise==True:
    df_train['parent_ents']=df_train['parent_ents'].apply(lambda x: to_set(x))
    df_train['child_ents']=df_train['child_ents'].apply(lambda x: to_set(x))
    df_train['parent_ents']=df_train['parent_ents'].apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
    df_train['child_ents']=df_train['child_ents'].apply(lambda x:  [lemmatizer.lemmatize(i) for i in x])
    df_train['parent_ents']=df_train['parent_ents'].apply(lambda x: str(x))
    df_train['child_ents']=df_train['child_ents'].apply(lambda x: str(x))

    df_valid['parent_ents']=df_valid['parent_ents'].apply(lambda x: to_set(x))
    df_valid['child_ents']=df_valid['child_ents'].apply(lambda x: to_set(x))
    df_valid['parent_ents']=df_valid['parent_ents'].apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
    df_valid['child_ents']=df_valid['child_ents'].apply(lambda x:  [lemmatizer.lemmatize(i) for i in x])
    df_valid['parent_ents']=df_valid['parent_ents'].apply(lambda x: str(x))
    df_valid['child_ents']=df_valid['child_ents'].apply(lambda x: str(x))


    df_train['parent_nouns']=df_train['parent_nouns'].apply(lambda x: to_set(x))
    df_train['child_nouns']=df_train['child_nouns'].apply(lambda x: to_set(x))
    df_train['parent_nouns']=df_train['parent_nouns'].apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
    df_train['child_nouns']=df_train['child_nouns'].apply(lambda x:  [lemmatizer.lemmatize(i) for i in x])
    df_train['parent_nouns']=df_train['parent_nouns'].apply(lambda x: str(x))
    df_train['child_nouns']=df_train['child_nouns'].apply(lambda x: str(x))

    df_valid['parent_nouns']=df_valid['parent_nouns'].apply(lambda x: to_set(x))
    df_valid['child_nouns']=df_valid['child_nouns'].apply(lambda x: to_set(x))
    df_valid['parent_nouns']=df_valid['parent_nouns'].apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
    df_valid['child_nouns']=df_valid['child_nouns'].apply(lambda x:  [lemmatizer.lemmatize(i) for i in x])
    df_valid['parent_nouns']=df_valid['parent_nouns'].apply(lambda x: str(x))
    df_valid['child_nouns']=df_valid['child_nouns'].apply(lambda x: str(x))

    logging.info(f"Length after lemmatization (if applied) - df_train: {len(df_train)}, df_valid: {len(df_valid)}")

    #df_test['parent_ents']=df_test['parent_ents'].apply(lambda x: to_set(x))
    #df_test['child_ents']=df_test['child_ents'].apply(lambda x: to_set(x))
    #df_test['parent_ents']=df_test['parent_ents'].apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
    #df_test['child_ents']=df_test['child_ents'].apply(lambda x:  [lemmatizer.lemmatize(i) for i in x])
    #df_test['parent_ents']=df_test['parent_ents'].apply(lambda x: str(x))
    #df_test['child_ents']=df_test['child_ents'].apply(lambda x: str(x))

  # subset train and validation dataset with target entities
  users_train = set(list(train_pos_edges.keys()) + list(train_neg_edges.keys()))
  users_eval = set(list(eval_pos_edges.keys()) + list(eval_neg_edges.keys()))
  #users_test = set(list(test_pos_edges.keys()) + list(test_neg_edges.keys()))






  print('ENTS DATASET', ents_dataset)
  print('NOUNS DATASET', nouns_dataset)
    




  logging.info(f"Filtering df_train using entities and nouns...")

    # Custom function allowing to decide whether user wants to utilize independency between nouns and entities in dataset filtering
  print("Filtering for entities...")

  if ents_dataset:
    df_train_sub = df_train[
        (df_train['child_ents'].str.contains('|'.join(ents_dataset))) &
        (df_train['parent_ents'].str.contains('|'.join(ents_dataset)))
    ].reset_index(drop=True)

    df_valid_sub = df_valid[
        (df_valid['child_ents'].str.contains('|'.join(ents_dataset))) &
        (df_valid['parent_ents'].str.contains('|'.join(ents_dataset)))
    ].reset_index(drop=True)
  else:
    df_train_sub = df_train.copy()
    df_valid_sub = df_valid.copy()

  
  if nouns_dataset:
    df_train_noun_filtered = df_train[
        (df_train['child_nouns'].str.contains('|'.join(nouns_dataset))) &
        (df_train['parent_nouns'].str.contains('|'.join(nouns_dataset)))
    ].reset_index(drop=True)

    df_valid_noun_filtered = df_valid[
        (df_valid['child_nouns'].str.contains('|'.join(nouns_dataset))) &
        (df_valid['parent_nouns'].str.contains('|'.join(nouns_dataset)))
    ].reset_index(drop=True)

    if not ents_dataset:
        df_train_sub = df_train_noun_filtered
        df_valid_sub = df_valid_noun_filtered
    else:
        if independent_filtering:
            # Independent filtering 
            df_train_sub = pd.concat([df_train_sub, df_train_noun_filtered]).drop_duplicates().reset_index(drop=True)
            df_valid_sub = pd.concat([df_valid_sub, df_valid_noun_filtered]).drop_duplicates().reset_index(drop=True)
        else:
            # Allowing for mixed entity-noun pairs
            df_train_sub = pd.concat([df_train_sub, df_train_noun_filtered,
                df_train[(df_train['child_nouns'].str.contains('|'.join(nouns_dataset))) &
                         (df_train['parent_ents'].str.contains('|'.join(ents_dataset))) |
                         (df_train['child_ents'].str.contains('|'.join(ents_dataset))) &
                         (df_train['parent_nouns'].str.contains('|'.join(nouns_dataset)))]]).drop_duplicates().reset_index(drop=True)

            df_valid_sub = pd.concat([df_valid_sub, df_valid_noun_filtered,
                df_valid[(df_valid['child_nouns'].str.contains('|'.join(nouns_dataset))) &
                         (df_valid['parent_ents'].str.contains('|'.join(ents_dataset))) |
                         (df_valid['child_ents'].str.contains('|'.join(ents_dataset))) &
                         (df_valid['parent_nouns'].str.contains('|'.join(nouns_dataset)))]]).drop_duplicates().reset_index(drop=True)

  post_filter_train = len(df_train_sub)
  post_filter_valid = len(df_valid_sub)

  print(f"Train set length after frequency and similarity filtering: {post_filter_train}")
  print(f"Valid set length after frequency and similarity filtering: {post_filter_valid}")

  logging.info(f"Filtered df_train from {len(df_train)} to {post_filter_train} rows")
  logging.info(f"Filtered df_valid from {len(df_valid)} to {post_filter_valid} rows")






    

  print("Final df_train_sub head:")
  print(df_train_sub.head())
  print("Final df_valid_sub head:")
  print(df_valid_sub.head())




  if full_agreement:
    logging.info(f"Applying full agreement filtering...")
    pre_agreement_train = len(df_train_sub)
    df_train_sub = df_train_sub[df_train_sub['agreement_fraction'] == 1].reset_index(drop=True)
    post_agreement_train = len(df_train_sub)
    logging.info(f"Filtered df_train_sub by agreement_fraction from {pre_agreement_train} to {post_agreement_train} rows")

    pre_agreement_valid = len(df_valid_sub)
    df_valid_sub = df_valid_sub[df_valid_sub['agreement_fraction'] == 1].reset_index(drop=True)
    post_agreement_valid = len(df_valid_sub)
    logging.info(f"Filtered df_valid_sub by agreement_fraction from {pre_agreement_valid} to {post_agreement_valid} rows")
  #script_directory = os.path.dirname(os.path.abspath(__file__))
  #output_path = os.path.join(script_directory, 'df_train_sub.csv')
  #df_train_sub.to_csv(output_path, index=False)
  #logging.info(f"df_train_sub has been saved to {output_path}")



  # compute class weights
  y = df_train_sub['label'].values
  y2 = df_valid_sub['label'].values
  #y3 = df_test_sub['label'].values
  classes_array = np.array([0, 1, 2])  
  class_weights=compute_class_weight('balanced', classes = classes_array , y = y)
  class_weights2=compute_class_weight('balanced', classes = classes_array , y = y2)
  #class_weights3=compute_class_weight('balanced', classes = classes_array , y = y3)
  print('WEIGHTS_train', class_weights)
  print('WEIGHTS_valid', class_weights2)
  #print('WEIGHTS_test', class_weights3)
  class_weights=torch.tensor(class_weights,dtype=torch.float)
  return df_train_sub, df_valid_sub, train_pos_edges, train_neg_edges, eval_pos_edges, eval_neg_edges, class_weights


def multi_acc(y_pred, y_true):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    correct_pred = (y_pred_tags == y_true).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    return acc

def create_author_vector(author, pos_edges, neg_edges):
    pos_vector = sum(pos_edges.get(author, {}).values())
    neg_vector = sum(neg_edges.get(author, {}).values())
    return np.array([pos_vector, neg_vector])

def sensitivity_correlation_analysis(df, pos_edges, neg_edges):
    user_vectors = {}
    unique_users = set(df['author_parent'].unique()) | set(df['author_child'].unique())
    for user in unique_users:
        user_vectors[user] = create_author_vector(user, pos_edges, neg_edges)

    cosine_similarities = []
    labels = []

    for _, row in df.iterrows():
        parent_vector = user_vectors[row['author_parent']]
        child_vector = user_vectors[row['author_child']]
        
        if np.any(parent_vector) and np.any(child_vector):  # Avoiding division by zero
            similarity = 1 - cosine(parent_vector, child_vector)
        else:
            similarity = 0  
        
        cosine_similarities.append(similarity)
        labels.append(row['label'])

    tau, p_value = kendalltau(cosine_similarities, labels)
    return tau, p_value






def get_top_100_entities(df):
    all_entities = []
    for ents in df['parent_ents'].apply(literal_eval):
        all_entities.extend(ents)
    for ents in df['child_ents'].apply(literal_eval):
        all_entities.extend(ents)
    
    entity_counts = Counter(all_entities)
    top_100 = entity_counts.most_common(100)
    
    print("Top 100 most common entities:")
    for entity, count in top_100:
        print(f"{entity}: {count}")
    
    return top_100



def process_batch(batch, nlp):
    nouns = []
    for text in batch:
        doc = nlp(text)
        nouns.extend([token.text.lower() for token in doc if token.pos_ == "NOUN"])
    return nouns



def get_top_n_nouns(df, N=300, batch_size=1000):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  
    
    all_texts = pd.concat([df['body_parent'], df['body_child']]).tolist()
    total_texts = len(all_texts)
    
    all_nouns = []
    
    # Create a progress bar
    pbar = tqdm(total=total_texts, desc="Processing texts")
    
    pbar.close() 
    
    noun_counts = Counter(all_nouns)
    top_n_nouns = noun_counts.most_common(N)
    
    
    total_noun_occurrences = sum(noun_counts.values())
    
    
    top_n_occurrences = sum(count for _, count in top_n_nouns)
    coverage_percentage = (top_n_occurrences / total_noun_occurrences) * 100
    
    print(f"Top {N} most common nouns and their counts:")
    for noun, count in top_n_nouns:
        print(f"{noun}: {count}")
    
    print(f"Coverage of top {N} nouns: {coverage_percentage:.2f}%")
    
    return top_n_nouns


#function utilised for merging overlapping edges
def merge_edges(entity_edges, noun_edges):
    merged = {}
    for user in set(entity_edges.keys()) | set(noun_edges.keys()):
        merged[user] = {**noun_edges.get(user, {}), **entity_edges.get(user, {})}
    return merged



def save_noun_edges(pos_edges, neg_edges, filename):
    combined_edges = {}
    for edges in [pos_edges, neg_edges]:
        for user, items in edges.items():
            if user not in combined_edges:
                combined_edges[user] = {}
            combined_edges[user].update(items)
    
    with open(filename, 'w') as f:
        json.dump(combined_edges, f)



def analyze_noun_distribution(noun_freq_dict, top_n=1000):
    sorted_nouns = sorted(noun_freq_dict.items(), key=lambda x: x[1], reverse=True)
    frequencies = [freq for _, freq in sorted_nouns[:top_n]]
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(frequencies) + 1), frequencies)
    plt.title(f'Frequency Distribution of Top {top_n} Nouns')
    plt.xlabel('Noun Rank')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)
    plt.savefig('noun_distribution.png')
    plt.close()
    
    return sorted_nouns

def calculate_cumulative_coverage(sorted_nouns):
    total_occurrences = sum(freq for _, freq in sorted_nouns)
    cumulative_coverage = []
    current_sum = 0
    
    for i, (_, freq) in enumerate(sorted_nouns, 1):
        current_sum += freq
        coverage = current_sum / total_occurrences
        cumulative_coverage.append((i, coverage))
    
    plt.figure(figsize=(12, 6))
    plt.plot([x[0] for x in cumulative_coverage], [x[1] for x in cumulative_coverage])
    plt.title('Cumulative Coverage of Nouns')
    plt.xlabel('Number of Nouns')
    plt.ylabel('Cumulative Coverage')
    plt.xscale('log')
    plt.grid(True)
    plt.savefig('cumulative_coverage.png')
    plt.close()
    
    return cumulative_coverage

from kneed import KneeLocator

def find_elbow_point(coverage_data):
    x = [point[0] for point in coverage_data]
    y = [point[1] for point in coverage_data]
    
    kneedle = KneeLocator(x, y, S=1.0, curve='concave', direction='increasing')
    elbow_point = kneedle.elbow
    
    plt.figure(figsize=(12, 6))
    plt.plot(x, y)
    plt.vlines(elbow_point, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors='r')
    plt.title('Cumulative Coverage with Elbow Point')
    plt.xlabel('Number of Nouns')
    plt.ylabel('Cumulative Coverage')
    plt.xscale('log')
    plt.grid(True)
    plt.savefig('elbow_point.png')
    plt.close()
    
    return elbow_point




def save_filtered_datasets(df_train_sub, df_valid_sub, train_length_expected=2071, valid_length_expected=247):

    
    assert len(df_train_sub) == train_length_expected, f"Train dataset length is {len(df_train_sub)}, expected {train_length_expected}"
    assert len(df_valid_sub) == valid_length_expected, f"Validation dataset length is {len(df_valid_sub)}, expected {valid_length_expected}"

   
    df_train_sub.to_csv('filtered_train_dataset.csv', index=False)
    df_valid_sub.to_csv('filtered_valid_dataset.csv', index=False)
    
    print(f"Filtered train dataset saved to 'filtered_train_dataset.csv' with {len(df_train_sub)} rows")
    print(f"Filtered validation dataset saved to 'filtered_valid_dataset.csv' with {len(df_valid_sub)} rows")


def save_noun_edges(pos_edges, neg_edges, filename):
    combined_edges = {}
    for edges in [pos_edges, neg_edges]:
        for user, items in edges.items():
            if user not in combined_edges:
                combined_edges[user] = {}
            combined_edges[user].update(items)
    
    with open(filename, 'w') as f:
        json.dump(combined_edges, f)