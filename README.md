# extended-word2vec
Trains source-aware representations of word embeddings. 


Stages: Data Collection, Data Processing, Training Embeddings and finally Sparse Representation & testing.

## 1- Data collection:

- Run `giga_parser.py` to parse SGML format of Gigaword:
-- expects .gz files to be already there
-- you just need to pass a file with paths to the .gz files

- Run `download_cc.py` to download articles from gigaword
-- WARC format, then it's processed to text within the same script
-- Expects a file with json objects returned from commoncrawl's CDX API 


## Data Processing

- Run `process_corpus.py` for tokenization, cleaning weird characters and other noisy data, produces new tokenized text files
- Run `NER.java` for named entity recognition and replacement of entity to entity#media_source, produces new tagged text files

## Training Embeddings

- Run `create_meta.py` to create meta files (vocab, entities, .. etc.) used by the model
- Run `main.py` to train word embeddings, you need to specify path to output files
- You can use `vis.py` to visualize embeddings whether by PCA or t-SNE

## Sparse Representation & testing

- Finally run `sparse_and_test.py`, produces classification resutls directly
