import sys

from sklearn.metrics.pairwise import cosine_similarity
import json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup


def format_ref(r):
    return '\n'.join(['{0}-{1}:{2}'.format(x['start'],x['end'],x['ref_id']) for x in r])


def format_author(author):    
    return " ".join([author['first'], " ".join(author['middle']), author['last']])


def json_reader(file):
    '''takes a json file, processes the body, ref, and bib data into a dataframe
    heavily inspired by work done by Kaggle user xhlulu: https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv
    with open(file) as f:
        j = json.load(f)
        
    #format the body text so the sections are clear, but it's easy to view the whole thing
    body_text = '\n\n'.join(['<section {}> '.format(n) + x['section'] + '\n\n' + x['text'] for n,x in enumerate(j['body_text'])])
    ref_spans = '\n\n'.join(['<section {}> '.format(n) + x['section'] + '\n\n' + format_ref(x['ref_spans']) for n,x in enumerate(j['body_text'])])
    cite_spans = '\n\n'.join(['<section {}> '.format(n) + x['section'] + '\n\n' + format_ref(x['cite_spans']) for n,x in enumerate(j['body_text'])])
    
    #format references in a similar way
    ref_data = '\n\n'.join([k + '\n\n' + v['text'] + '\n\nlatex- {}'.format(v['latex']) for k,v in j['ref_entries'].items()])

    #put the bibliography together, and format the authors
    for k in j['bib_entries']:
        j['bib_entries'][k]['author_list'] = ', '.join([format_author(a) for a in (j['bib_entries'][k]['authors'])])

    bib_keys = ['ref_id', 'title', 'author_list', 'year', 'venue', 'volume', 'issn', 'pages', 'other_ids']
    bib_data = '\n\n'.join([', '.join([str(x[k]) for k in bib_keys]) for _,x in j['bib_entries'].items()])
    df = pd.DataFrame(index=[0], data={'body_text':body_text, 
                                            'cite_spans':cite_spans, 
                                            'ref_spans':ref_spans,
                                            'ref_data': ref_data,
                                            'bib_data': bib_data,
                                            'paper_id': j['paper_id']})
    
    return df


def parse_folder(data_folder):
    filelist = glob('/kaggle/input/CORD-19-research-challenge/{0}/{0}/*'.format(data_folder))
    filelist.sort()
    print('{} has {} files'.format(data_folder, len(filelist)))

    df_ls=[]
    for n,file in enumerate(filelist):
        if n%1000==0:
            print(n,file[-46:])
        df = json_reader(file)
        df_ls.append(df)
    return pd.concat(df_ls)


def load_meta():
    meta = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
    meta.rename(columns={'sha':'paper_id'}, inplace=True)
    return meta


#going through each of the four folders of json files and put everything into one dataframe
#takes around 3-4min to complete
def combine_datasets():
    df_ls = []
    for folder in ['comm_use_subset', 'noncomm_use_subset', 'custom_license', 'biorxiv_medrxiv']:
        t = parse_folder(folder)
        df_ls.append(t)
    df = pd.concat(df_ls)
    
    meta = load_meta()
    df = meta.merge(df, on='paper_id', how='left')
    return df


def get_doc_vec(tokens):
    #combine scispacy word embeddings from a document into a single document vector
    #filter out any stop words like 'the', and remove any punction/numbers
    w_all = np.zeros(tokens[0].vector.shape)
    n=0
    for w in tokens:
        if (not w.is_stop) and (len(w)>1) and (not w.is_punct) and (not w.is_digit):
            w_all += w.vector
            n+=1
    return (w_all / n) if n>0 else np.zeros(tokens[0].vector.shape)


def process_all_docs(col,id_col):
    vecs = {}
    for n,row in df.iterrows():
        if n%5000==0:
            print(n)
        if len(row[col]) > 0:
            vecs[row[id_col]] = get_doc_vec(nlp(row[col]))
    return vecs


def get_matching_papers(q_str, sent_df):
    q_nlp = nlp(q_str)
    
    #use nouns and objects from the question to find keywords and phrases
    noun_ls = []
    for noun in q_nlp.noun_chunks:
        if ('obj' in noun.root.dep_ or 'subj' in noun.root.dep_ or noun.root.dep_ == 'appos') and len(str(noun.root)) > 1 and noun.root.is_stop == False:
            noun_ls.append(str(noun.root).lower())
    
    #also use any entities found in the text. Don't take the root of these
    noun_ls += [x.text for x in q_nlp.ents]
    noun_ls = set(noun_ls)
    print('keywords : {}'.format(noun_ls))
    key_condition = match_df['text'].str.contains(r'|'.join(noun_ls))

    #get similarity to all available sentences and make a dataframe with the sentences
    sent_sims = cosine_similarity(sent_df['vecs'].tolist(), model.encode([q_str]))
    sent_df['score'] = sent_sims.max(axis=1)
    
    #filter out sentences that don't belong to papers which include the keywords
    #sent_df = sent_df[sent_df['title'].isin(match_df[key_condition]['title'])]
    
    #filter out sentences with a low match score
    sent_df = sent_df[sent_df['score']>SUBMATCH_THRESHOLD*sent_df['score'].max()]

    #sort by papers with a high number of relevant sentences and return the results
    return sent_df.groupby('title').agg({'sent':'count', 'score':'mean'}).sort_values(by='sent', ascending=False).reset_index().rename(columns={'sent':'relevant sentences'})


def get_text(url,abstract=False,body=True,bib=False):
    '''PROVIDED BY DR. LEVINE AT ACCENTURE
    Returns the full text of a paper, given the source html, provided the paper is in the rough format of the Wiley Online Library
    Ex: https://asistdl.onlinelibrary.wiley.com/doi/10.1002/asi.24357
    abstract = True will return abstract as part of the text
    body = True will return the body of the paper as part of the text
    bib = True will return the bibliography as part of the text

    Author: Aaron Levine
    '''
    text = ''
    try:
        html_text = !wget -qO- --timeout=180 $url
        soup = BeautifulSoup('\n'.join(html_text),'html.parser')
        if abstract:
            abstract_txt = '\n'.join([x.text for x in soup.find('div',{'class':'abstract-group'}).findChildren(recursive=False)])
            text += abstract_txt
        if body:
            body_txt = '\n'.join([x.text for x in soup.find('section',{'class':'article-section article-section__full'}).findChildren(recursive=False) if True not in [tag.has_attr('data-bib-id') for tag in x.find_all()]])
            text += body_txt
        if bib:
            bib_txt = '\n'.join([x.text for x in soup.find('section',{'class':'article-section article-section__full'}).findChildren(recursive=False) if True in [tag.has_attr('data-bib-id') for tag in x.find_all()]])
            text += bib_txt
    except:
        print('failed to load paper')

    if len(text) > 0:
        print('found the paper!')
    return text
