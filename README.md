# DFHD
## Folder Specification
- data/ folder contains necessary data or scripts for generating data.
  - drug-atc.csv, ndc2atc_level4.csv, ndc2rxnorm_mapping.txt, atc2rxnorm.pkl: mapping files that collected from external sources
  - idx2SMILES.pkl: Drug ID to drug SMILES string dictionary
  - ddi_mask_H.pkl: A mask matrix containing the relations between molecule and substructures
  - substructure_smiles.pkl: A list containing the smiles of all the substructures.
  - ddi_mask_H.py: is used to get ddi_mask_H.pkl
  - processing.py: is used to process the MIMIC original dataset
- Under MIMIC Dataset policy, we are not allowed to distribute the datasets. Practioners could go to https://physionet.org/content/mimiciii/1.4/ and requrest the access to MIMIC-III dataset and then run our processing script to get the complete preprocessed dataset file.

- src/ folder contains all the source code.
  - modules/: Code for model definition.
  - utils.py: Code for metric calculations and some data preparation.
  - training.py: Code for the functions used in training and evaluation.
  - main.py: Train or evaluate our DFHD Model.

## STEP1:Data Processing
  - Go to https://physionet.org/content/mimiciii/1.4/ to download the MIMIC-III dataset
  - go into the folder and unzip three main files (PROCEDURES_ICD.csv.gz, PRESCRIPTIONS.csv.gz, DIAGNOSES_ICD.csv.gz)
  - run processing.py to get a complete records_final.pkl
  ''' python processing.py '''

## STEP2:Package Dependency
  python 3.8.5, scipy 1.10.1, pandas 1.1.3, torch 1.12.1, numpy 1.19.2, dill, rdkit 

## STEP3:Run Model
  ''' python main.py '''
