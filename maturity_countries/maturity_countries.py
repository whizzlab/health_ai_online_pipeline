# Imports
import base64

import pandas as pd
import numpy as np
from tqdm import trange, tqdm
from datetime import datetime as dt
import sys
from pathlib import Path

import tensorflow as tf
import tensorflow_text as text

from google.cloud import bigquery
from google.cloud import storage
from pandas_gbq import schema

from flashgeotext.geotext import GeoText
from geopy.geocoders import Nominatim

# Define gloubal variables
project = '' # project ID
mature_model = None
characteristics_model = None
geolocator = Nominatim(user_agent='health_ai_abstract_scraper')
geotext = GeoText()

# Define functions

def hello_pubsub(event, context): # Cloud function entry

    pubsub_message = base64.b64decode(event['data']).decode('utf-8')
    
    df = pd.read_json(pubsub_message, convert_dates = True, orient = 'table')

    label_included(df)

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

def find_affil_countries(affils: list, retry_count = 5):
    
    if (affils == affils) and (affils != None): # Check to make sure not NaN or no affils listed
        country_list = []
        location = None
        
        last_affil = None
        last_country = None
        
        try_count = 0
    
        for affil in affils:
            if affil == last_affil: # Check to see if we've seen this before and take a shortcut if we have
                country_list = country_list + last_country

            else:
                while try_count < retry_count:
                    try:
                        last_affil = affil # Set that we've examined this affil
                
                        countries = [*geotext.extract(input_text=affil, span_info=True)['countries'].keys()] # Look for countries
                
                        if (len(countries) == 0): # If we dont find a country look harder
                            cities = [*geotext.extract(input_text=affil, span_info=True)['cities'].keys()]
                            if len(cities) > 0:
                                location = geolocator.geocode(cities[-1])
                            else:
                                location = geolocator.geocode(' '.join(affil.split(" ")[-2:]))
                                if location == None:
                                    location = geolocator.geocode(affil)
                                    if location == None:
                                        tqdm.write("Can't find a country for:")
                                        tqdm.write(affil)
                                        country_list = country_list + [np.nan]
                                        last_country = [np.nan]
                            
                        else: # If we do find a country then att it to the list and set the last_country variable
                            country_list = country_list + countries
                            last_country = countries
        
                        if location != None: # If we found an address using the other search techniques
                            countries = [*geotext.extract(input_text=location.address, span_info=True)['countries'].keys()]
                            country_list = country_list + countries
                            last_country = countries
                            
                        break
                            
                    except:
                        try_count += 1
                        tqdm.write(f"Error parsing {affil}, trying again for a maximum of 5 times.")
                        last_country = [np.nan]
                        
    else: # If the affil is NaN then make the country list NaN
        country_list = [np.nan]
        
    unique_countries = list(set(country_list))
    first_affil_country = country_list[0]
    last_affil_country = country_list[-1]    
        
    return country_list, unique_countries, first_affil_country, last_affil_country

def parse_affil_countries(df, max_consecutive_failures = 5, filter_column = 'include'):
    
    consecutive_failures = 0
    
    country_df = df.copy()
    
    country_df['affil_countries'] = np.nan
    country_df['affil_countries_unique'] = np.nan
    country_df['affil_first_country'] = np.nan
    country_df['affil_last_country'] = np.nan
    
    with tqdm(total=country_df[country_df[filter_column] == 1].shape[0], file=sys.stdout) as pbar:
        for row in country_df[country_df[filter_column] == 1].itertuples():

            try:
                affils = row.author_affils
    
                country_list, unique_countries, first_affil_country, last_affil_country = find_affil_countries(affils)
    
                country_df.loc[row.Index, 'affil_countries'] = str(country_list)
                country_df.loc[row.Index, 'affil_countries_unique'] = str(list(set(country_list)))
                country_df.loc[row.Index, 'affil_first_country'] = country_list[0]
                country_df.loc[row.Index, 'affil_last_country'] = country_list[-1]
            
                consecutive_failures = 0
        
            except Exception as e:
                tqdm.write(str(e))
                consecutive_failures += 1
                if consecutive_failures >=  max_consecutive_failures:
                    tqdm.write("Failed too many in a row, something is broken, stopping and returning possibly partially labelled DF...")
                    break
                
            pbar.update(1)
            
    country_df.replace("[nan]", np.nan, inplace=True)
            
    return country_df

def label_included(article_df):

    # GBQ setup
    project = 'health-ai-320507'
    client = bigquery.Client(project=project)

    # Classify mature studies
    print("Classifying included articles for mature studies...")
    
    global mature_model
    
    if not mature_model:
        download_model(prefix='maturity_bert/')
        mature_model = tf.saved_model.load('/tmp/maturity_bert')
    
    mature_labelled = predict_labels(model = mature_model,
                                          df = article_df[article_df.include == 1],
                                          label_name = 'mature')
    
    article_df.loc[mature_labelled.index, 'mature'] = mature_labelled['mature']
    
    article_df.mature = article_df.mature.fillna(0).astype(int)
    
    print(f"{article_df.mature.sum()} articles labelled mature.")
    
    # Label study characteristics
    global characteristics_model
    
    char_labels = ['algo_neural_net', 'algo_support_vector', 'algo_regression', 'algo_decision_tree', 'feat_xr', 'feat_ct', 'feat_mri', 'feat_eeg',
 'feat_ecg', 'feat_emg', 'feat_us', 'feat_echo', 'feat_histo', 'feat_oct', 'feat_mamm', 'feat_endoscop', 'feat_gene', 'feat_bio', 'feat_nlp', 'feat_ehr',
 'subspec_icu', 'subspec_ed', 'spec_paeds', 'spec_id', 'subspec_sepsis', 'subspec_cov19', 'subspec_dermca', 'spec_onc', 'subspec_lungca', 'subspec_brainca',
 'subspec_gica', 'subspec_hepca', 'subspec_prosca', 'subspec_gynonc', 'subspec_haemonc', 'subspec_breastca', 'subspec_urology', 'spec_psych',
 'spec_msk', 'spec_gi', 'spec_hep', 'spec_resp', 'subspec_pneum', 'spec_neuro', 'subspec_epilep', 'subspec_cva', 'subspec_alzh', 'spec_cvs',
 'subspec_ihd', 'subspec_hf', 'subspec_arrhyt', 'spec_dm', 'subspec_retina', 'spec_haem', 'spec_obs', 'spec_renal']
    
    if not characteristics_model:
        download_model(prefix='multilabel_characteristics_bert/')
        characteristics_model = tf.saved_model.load('/tmp/multilabel_characteristics_bert')
        
    characteristics_labelled = predict_labels(model = characteristics_model,
                                              df = article_df[article_df.include == 1],
                                              label_name = char_labels,
                                              logit_model=False)
    
    article_df.loc[characteristics_labelled.index, char_labels] = characteristics_labelled[char_labels]
    
    print("Characteristic labelling complete...")

    # Extract countries from author affiliations
    print("Parsing countries from author affiliations...")
    
    country_parsed = parse_affil_countries(article_df)

    ## using first affiliation as primary country
    ## fill first with data from last author, then with pubmed country metadata
    country_parsed['affil_fill_country'] = country_parsed['affil_first_country']

    country_parsed['affil_fill_country'] = country_parsed['affil_first_country'].fillna(country_parsed['affil_last_country'])
    country_parsed['affil_fill_country'] = country_parsed['affil_first_country'].fillna(country_parsed['journal_country'])

    country_parsed['affil_first_country'] = country_parsed['affil_first_country'].astype('string')

    ##clean countries
    country_parsed["affil_first_country"].replace({"England": "United Kingdom", 
                                                   "Wales": "United Kingdom", 
                                                   "Scotland": "United Kingdom", 
                                                   "China (Republic : 1949- )" : "Taiwan"}, inplace=True)    

    ##lowercasing list of ANY author
    country_parsed['countries_lc'] = country_parsed['affil_countries_unique'].astype(str).str.lower()

    country_parsed['countries_lc'] = country_parsed['countries_lc'].fillna('')  

    lmic_list = ["afghanistan", "burundi", "burkina faso", "central african republic", "congo", "eritrea", 
             "ethiopia", "guinea", "gambia", "guinea-bissau", "liberia", "madagascar", "mali", "mozambique", "malawi", 
             "niger", "north korea", "democratic republic of korea", "rwanda", "sudan", "sierra leone", "somalia", "south sudan", "syrian arab republic", 
             "chad", "togo", "uganda", "yemen", "angola", "benin", "bangladesh", "belize", "bolivia", "bhutan", 
             "cote d'ivoire", "ivory coast", "cameroon", "congo", "comoros", "cabo verde", "djibouti", "algeria", "egypt", 
             "micronesia", "ghana", "honduras", "haiti", "indonesia", "india", "iran", "kenya", 
             "kyrgyz republic", "cambodia", "kiribati", "lao", "sri lanka", "lesotho", "morocco", "myanmar", "mongolia", 
             "mauritania", "nigeria", "nicaragua", "nepal", "pakistan", "philippines", "papua new guinea", 
             "west bank and gaza", "palestinbe", "senegal", "solomon islands", "el salvador", "sao tome", "eswatini", 
             "tajikistan", "timor-leste", "tunisia", "tanzania", "ukraine", "uzbekistan", "vietnam", "vanuatu", "samoa", 
             "zambia", "zimbabwe", "albania", "argentina", "armenia", "american samoa", "azerbaijan", "bulgaria", 
             "bosnia", "belarus", "brazil", "botswana", "china", "colombia", "costa rica", "cuba", 
             "dominica", "dominican republic", "ecuador", "fiji", "gabon", "georgia", "equatorial guinea", "grenada", 
             "guatemala", "guyana", "iraq", "jamaica", "jordan", "kazakhstan", "lebanon", "libya", "lucia", "moldova", 
             "maldives", "mexico", "marshall islands", "north macedonia", "montenegro", "mauritius", "malaysia", "namibia", 
             "panama", "peru", "paraguay", "romania", "russian federation", "russia", "serbia", "suriname", "thailand", "turkmenistan", 
             "tonga", "turkey", "tuvalu", "st. vincent", "grenadines", "kosovo", "south africa", "venezuela"]

    lmic_lower_list = ["afghanistan", "burundi", "burkina faso", "central african republic", "congo", "eritrea", 
             "ethiopia", "guinea", "gambia", "guinea-bissau", "liberia", "madagascar", "mali", "mozambique", "malawi", 
             "niger", "north korea", "democratic republic of korea", "rwanda", "sudan", "sierra leone", "somalia", "south sudan", "syrian arab republic", 
             "chad", "togo", "uganda", "yemen", "angola", "benin", "bangladesh", "belize", "bolivia", "bhutan", 
             "cote d'ivoire", "ivory coast", "cameroon", "congo", "comoros", "cabo verde", "djibouti", "algeria", "egypt", 
             "micronesia", "ghana", "honduras", "haiti", "indonesia", "india", "iran", "kenya", 
             "kyrgyz republic", "cambodia", "kiribati", "lao", "sri lanka", "lesotho", "morocco", "myanmar", "mongolia", 
             "mauritania", "nigeria", "nicaragua", "nepal", "pakistan", "philippines", "papua new guinea", 
             "west bank and gaza", "palestinbe", "senegal", "solomon islands", "el salvador", "sao tome", "eswatini", 
             "tajikistan", "timor-leste", "tunisia", "tanzania", "ukraine", "uzbekistan", "vietnam", "vanuatu", "samoa", 
             "zambia", "zimbabwe"]

    #initiate
    country_parsed['lmic_author_flag'] = np.where(country_parsed['countries_lc'].str.contains('iran'), "1", "0")
    country_parsed['lmic_author_lower_flag'] = np.where(country_parsed['countries_lc'].str.contains('iran'), "1", "0")
    country_parsed['lmic_china_flag'] = np.where(country_parsed['countries_lc'].str.contains('china'), "1", "0")

    # Use lists
    for x in lmic_list:
        country_parsed['lmic_author_flag'] = np.where(country_parsed['countries_lc'].str.contains(x), "1", country_parsed['lmic_author_flag'])
    
    for y in lmic_lower_list:
        country_parsed['lmic_author_lower_flag'] = np.where(country_parsed['countries_lc'].str.contains(x), "1", country_parsed['lmic_author_lower_flag'])

    # Correct dtypes
    country_parsed['lmic_author_flag'] = country_parsed['lmic_author_flag'].astype(float)
    country_parsed['lmic_author_lower_flag'] = country_parsed['lmic_author_lower_flag'].astype(float)
    country_parsed['lmic_china_flag'] = country_parsed['lmic_china_flag'].astype(float)

    # Add a year column
    country_parsed['year'] = country_parsed.pubmed_date.dt.year

    # Generate schema to dump into BQ
    schema_articles = schema.generate_bq_schema(country_parsed)['fields']
    
    # Push the articledf to a temporary table
    print("Adding data to temporary BQ table...")
    country_parsed.to_gbq(destination_table = 'health_ai_abstracts.ai_abstracts_scrape_temp',
                        project_id = project,
                        if_exists = 'replace', 
                        table_schema = schema_articles,
                        chunksize=500)
    
    # Merge query to insert new articles into main table and avoid duplicates
    
    print("Merging temporary table with main BQ table...")
    
    merge_query = """
    MERGE health_ai_abstracts.ai_abstracts_scrape AS target
    USING health_ai_abstracts.ai_abstracts_scrape_temp AS source
    ON target.pmid = source.pmid
    WHEN NOT MATCHED BY target THEN
        INSERT(pmid, doi, title, abstract, article_date, pubmed_date, year, article_type, lang, journal, journal_short, journal_country, authors, author_affils, keywords, mesh_terms, references_pmids,
        include, mature,
        affil_countries, affil_countries_unique, affil_first_country, affil_last_country, affil_fill_country, countries_lc, lmic_author_flag, lmic_author_lower_flag, lmic_china_flag,
        algo_neural_net, algo_support_vector, algo_regression, algo_decision_tree, feat_xr, feat_ct, feat_mri, feat_eeg,
        feat_ecg, feat_emg, feat_us, feat_echo, feat_histo, feat_oct, feat_mamm, feat_endoscop, feat_gene, feat_bio, feat_nlp, feat_ehr,
        subspec_icu, subspec_ed, spec_paeds, spec_id, subspec_sepsis, subspec_cov19, subspec_dermca, spec_onc, subspec_lungca, subspec_brainca,
        subspec_gica, subspec_hepca, subspec_prosca, subspec_gynonc, subspec_haemonc, subspec_breastca, subspec_urology, spec_psych,
        spec_msk, spec_gi, spec_hep, spec_resp, subspec_pneum, spec_neuro, subspec_epilep, subspec_cva, subspec_alzh, spec_cvs,
        subspec_ihd, subspec_hf, subspec_arrhyt, spec_dm, subspec_retina, spec_haem, spec_obs, spec_renal
        )
        
        VALUES(pmid, doi, title, abstract, article_date, pubmed_date, year, article_type, lang, journal, journal_short, journal_country, authors, author_affils, keywords, CAST(mesh_terms AS STRING), CAST(references_pmids AS STRING),
        include, mature,
        affil_countries, CAST(affil_countries_unique AS STRING), affil_first_country, CAST(affil_last_country AS STRING), affil_fill_country, countries_lc, lmic_author_flag, lmic_author_lower_flag, lmic_china_flag,
        algo_neural_net, algo_support_vector, algo_regression, algo_decision_tree, feat_xr, feat_ct, feat_mri, feat_eeg,
        feat_ecg, feat_emg, feat_us, feat_echo, feat_histo, feat_oct, feat_mamm, feat_endoscop, feat_gene, feat_bio, feat_nlp, feat_ehr,
        subspec_icu, subspec_ed, spec_paeds, spec_id, subspec_sepsis, subspec_cov19, subspec_dermca, spec_onc, subspec_lungca, subspec_brainca,
        subspec_gica, subspec_hepca, subspec_prosca, subspec_gynonc, subspec_haemonc, subspec_breastca, subspec_urology, spec_psych,
        spec_msk, spec_gi, spec_hep, spec_resp, subspec_pneum, spec_neuro, subspec_epilep, subspec_cva, subspec_alzh, spec_cvs,
        subspec_ihd, subspec_hf, subspec_arrhyt, spec_dm, subspec_retina, spec_haem, spec_obs, spec_renal
        )
    """
    
    
    client.query(merge_query)
    
    print("Scraping complete, BQ tables updated.")