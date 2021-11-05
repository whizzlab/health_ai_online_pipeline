# Pubmed scraping and labelling of articles relating to the use of AI in human health

Code for the scraping and labelling of new abstracts relating to the use of AI in human health.

Abstracts are found and scraped using the Entrez API via biopython (https://github.com/biopython/biopython). Pre-trained models described .... (point to other Git of publication) are utilised to label scraped abstracts for inclusion (AI use relating to human health), being a mature study (comparing AI to the gold standard) and for characteristics. Author affiliation countries are parsed using the python packages flashgeotext (https://github.com/iwpnd/flashgeotext) and geopy (https://github.com/geopy/geopy)

These scripts are currently deployed within Google Cloud Platform as Cloud Functions, triggered once a day.
Due to the size of the models and time required to label the parsed abstracts, they are split into two separate functions which share data via GCP Pub/Sub.
