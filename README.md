# KGA_Bert
This project was for the Spoken Dialogue Systems class at Boise State University.
The project resulted in a research paper which was not submitted for publication.

Files of importance are:
  - src/KG/wikidata.py
  - src/training/train_aug_bert.ipynb
  - src/train/train_bert.ipynb

### wikidata.py
This script will precompute all KGEs for a given data set.
The exact process is as follows:
  1. Find all nouns within a data set
  2. Reduce to nouns with embeddings
  3. For all nouns, get their respective knowledge graph according to the algorithm outlined in the paper "KGA-BERT"
  4. Calculate each nouns KGE as outlined in the paper
  5. Output a csv file with every nouns KGE

### train_aug_bert.ipynb
This notebook walks through the training/evaluation process of the model.

### train_bert.ipynb
This notebook walks through the training/evaluation process of a simple BERT model.

To run any of these programs, make sure to pip install any dependencies. Special cases include: wikidata. You may need to rename wikidata.py seperately.

