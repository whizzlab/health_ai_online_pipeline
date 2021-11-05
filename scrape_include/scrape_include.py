# Imports
import base64

from Bio import Entrez
import pandas as pd
import numpy as np
from tqdm import trange, tqdm
from datetime import datetime as dt
import sys
from pathlib import Path

import tensorflow as tf
import tensorflow_text

from google.cloud import bigquery
from google.cloud import storage
from google.cloud import pubsub_v1

# Define global variables
project = 'health-ai-320507'
pubsub_topic = "maturity_country_gbq_trigger"
search_term = """((((((((["artificial intelligence") OR ("deep learning")) OR ("machine learning")) OR ("neural net")) OR ("transfer learning")) OR ("supervised learning")) OR (unsupervised learning)) ) OR (artificial intelligence[MeSH Terms])"""
include_model = None

# Define functions

def hello_pubsub(event, context): # Cloud function entry

    pubsub_message = base64.b64decode(event['data']).decode('utf-8')
    
    print(pubsub_message)
    print("Calling scraping and parsing functions...")

    scrape_include()

def search(query, retmax, mindate=None, maxdate=None, reldate=None):
    Entrez.email = 'swhebell@gmail.com'
    Entrez.api_key = '40ba1547513d683ed2f3d5adcbc26b1ad409'
    handle = Entrez.esearch(db='pubmed', 
                            retmax=retmax,
                            retmode='xml', 
                            term=query,
                            mindate=mindate,
                            maxdate=maxdate,
                            datetype='edat')
    results = Entrez.read(handle)
    return results

def fetch_details(id_list):
    ids = ','.join(id_list)
    Entrez.email = 'swhebell@gmail.com'
    Entrez.api_key = '40ba1547513d683ed2f3d5adcbc26b1ad409'
    handle = Entrez.efetch(db='pubmed',
                           retmode='xml',
                           id=ids)
    results = Entrez.read(handle)
    return results

def parse_article(article):
    # Empty dict for each article
    article_dict = {}
    
    # PMID
    article_dict['pmid'] = str(article['MedlineCitation']['PMID'])
    
    # Parse out the DOI, annoyingly it's mixed into PII fields and doesn't always seem to be there
    doi = np.nan
    
    for i in article['MedlineCitation']['Article']['ELocationID']:
        if i.attributes['EIdType'] == 'doi':
            doi = str(i)
        else:
            doi = np.nan
    
    article_dict['doi'] = doi
    
    # Title
    article_dict['title'] = article['MedlineCitation']['Article']['ArticleTitle']
    
    # Abstract
    try:
        article_dict['abstract'] = article['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
    except:
        article_dict['abstract'] = np.nan
    
    # Article dates
    try: # Doesn't always seem to have a date
        article_date= '-'.join(list(article['MedlineCitation']['Article']['ArticleDate'][0].values()))
        article_dict['article_date'] = dt.strptime(article_date, "%Y-%m-%d")
    except:
        pass
    
    # Date available on pubmed
    for i in article['PubmedData']['History']:
        if i.attributes['PubStatus'] == 'pubmed':
            pubmed_date = '-'.join(list(i.values())[:3])
            article_dict['pubmed_date'] = dt.strptime(pubmed_date, "%Y-%m-%d")
    
    # Article type
    try:
        article_dict['article_type'] = str(article['MedlineCitation']['Article']['PublicationTypeList'][0])
    except:
        pass
    
    # Article language
    try:
        article_dict['lang'] = article['MedlineCitation']['Article']['Language'][0]
    except:
        pass
    
    # Long form journal
    try:
        article_dict['journal'] = article['MedlineCitation']['Article']['Journal']['Title']
    except:
        pass
    
    # ISO Journal abbreviation
    try:
        article_dict['journal_short'] = article['MedlineCitation']['Article']['Journal']['ISOAbbreviation']
    except:
        pass
    
    # Journal country
    try:
        article_dict['journal_country'] = article['MedlineCitation']['MedlineJournalInfo']['Country']
    except:
        pass
    
    # Authors
    authors = []
    try: # Sometimes there aren't proper authors listed
        for author in article['MedlineCitation']['Article']['AuthorList']:
            authors.append(author['LastName'] + ' ' + author['ForeName'])
    except:
        authors = np.nan
    
    article_dict['authors'] = authors
    
    # Affiliations
    affils = []
    try:
        for author in article['MedlineCitation']['Article']['AuthorList']:
            affils.append(author['AffiliationInfo'][0]['Affiliation'])
    except:
        affils = np.nan
    
    article_dict['author_affils'] = affils
    
    # Article keywords
    try:
        article_dict['keywords'] = [str(i) for i in (article['MedlineCitation']['KeywordList'][0])]
    except:
        article_dict['keywords'] = np.nan
        

    # Article Mesh terms    
    mesh_terms = []
    try: # Not always mesh terms
        for i in article['MedlineCitation']['MeshHeadingList']:
            mesh_terms.append(str(i['DescriptorName']))
    except Exception as e:
        mesh_terms = np.nan
        
    article_dict['mesh_terms'] = mesh_terms
    
    # References (if included)
    references_pmids = []
    try: # References not always included
        for i in article['PubmedData']['ReferenceList'][0]['Reference']:
            references_pmids.append(str(i['ArticleIdList'][0]))
    except:
        references_pmids = np.nan
    
    article_dict['references_pmids'] = references_pmids
    
    return article_dict

def retrieve_articles(search_term, retmax, chunk_size = 50, mindate=None, maxdate=None):
    
    # Fetch a list of PMIDs from the search term
    result = search(search_term, retmax=retmax, mindate=mindate, maxdate=maxdate)
    id_list = result['IdList']
    
    search_time = dt.now()
    
    print(f"List of {len(id_list)} PMIDs retrieved of {result['Count']} results.")
    print("Downloading and parsing:")
    
    paper_list = []
    
    # Retrieve in chunks
    for chunk_i in trange(0, len(id_list), chunk_size):
        chunk = id_list[chunk_i:chunk_i + chunk_size]
    
        papers = fetch_details(chunk)
        
        for i, paper in enumerate(papers['PubmedArticle']):
            paper_list.append(parse_article(paper))
            
    df = pd.DataFrame(paper_list)
    
    df['pmid'] = df.pmid.astype(int)
    
    most_recent_date = df.pubmed_date.max()
                    
    return (df, search_time, most_recent_date)

def download_model(prefix:str):
    
    bucket_name = 'health_ai_models'
    dl_dir = '/tmp/'

    storage_client = storage.Client(project)
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)  # Get list of files
    
    # Iterate thorugh files and split out folders
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        file_split = blob.name.split("/")
        directory = "/".join(file_split[0:-1])
        Path(dl_dir + directory).mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(dl_dir + blob.name) 

def predict_labels(model, df, label_name: list, logit_model=True, feature_column='feature', chunk_size=100):
    
    labelled_df = df[feature_column].copy()
    labelled_df = labelled_df.to_frame()
    labelled_df[label_name] = np.nan
    
    with tqdm(total=len(labelled_df), file=sys.stdout) as pbar:
         for chunk_i in range(0, len(labelled_df.index), chunk_size):
                
                chunk = labelled_df.index[chunk_i:chunk_i + chunk_size]
                
                if logit_model:
                    labels = np.round(tf.sigmoid(model(tf.constant(labelled_df.loc[chunk, feature_column]))))
                else:
                    labels = labels = np.round(model(tf.constant(labelled_df.loc[chunk, feature_column])))

                labelled_df.loc[chunk, label_name] = labels
                
                pbar.update(len(chunk))
                
    return labelled_df

def scrape_include():
    
    # GBQ setup
    project = 'health-ai-320507'
    client = bigquery.Client(project=project)
    
    timing_query = """
                    SELECT max(pubmed_date)
                    FROM health_ai_abstracts.ai_abstracts_scrape
                    """
    
    # Query when the last search and most recent article were
    last_time = client.query(timing_query).result().to_dataframe().iloc[0,0].strftime("%Y/%m/%d")
    max_date = dt.now().strftime("%Y/%m/%d")
    
    
    # Retrieve data based on the date range specified
    print(f"Retrieving articles since {last_time}")
    
    article_df, search_time, most_recent_article_date = retrieve_articles(search_term = search_term,
                                                                           mindate = last_time,
                                                                           maxdate = max_date,
                                                                           retmax = 5000,
                                                                           chunk_size=50)
    
    # Create feature column
    article_df['feature'] = article_df['title'].astype(str) + ' ' + article_df['abstract'].astype(str)
    
    # Clean up a bit pre-labelling
    
    to_label = article_df.dropna(subset=['abstract'])
    
    try:
        to_label = to_label[~to_label['mesh_terms'].str.contains("'Mice|'Rat|'Dog|'Monkey|'Pig|'Cat|'Cow|'Horse|'Bird|'Fish", na = False)]
    except:
        print("No MeSH terms to clean animals out of.")

    to_label.dropna(subset=['abstract'], inplace=True)
    
    # Classification phase
   
    # Classify for inclusion
    print("Classifying articles for inclusion...")
    
    global include_model
    
    if not include_model: # Check if model already loaded, unlikely....
        download_model(prefix='ai_literature_bert')
        include_model = tf.saved_model.load('/tmp/ai_literature_bert') # Load model
    
    include_labelled = predict_labels(model = include_model,
                                      df = to_label,
                                      label_name = 'include')
    
    article_df.loc[include_labelled.index, 'include'] = include_labelled['include']
    
    article_df.include = article_df.include.fillna(0).astype(int)
    
    print(f"{article_df.include.sum()} articles labelled include.")

    # Push to pubsub for next phase of labelling
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project, pubsub_topic)

    data = article_df.to_json(orient='table')
    data = data.encode('utf-8')
    future = publisher.publish(topic_path, data)
    print(future.result())

    print("Pushed scraped and inclusion classified abstracts to pub/sub for next phase...")
    