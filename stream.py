import json
import numpy as np
import pandas as pd
from torch.utils import data
from sklearn.preprocessing import OneHotEncoder
from subword_nmt.apply_bpe import BPE
import codecs
from config import BIN_config_DBPE

vocab_path = './ESPF/protein_codes_uniprot.txt'
bpe_codes_protein = codecs.open(vocab_path)
pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
sub_csv = pd.read_csv('./ESPF/subword_units_map_uniprot.csv')

idx2word_p = sub_csv['index'].values
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

vocab_path = './ESPF/drug_codes_chembl.txt'
bpe_codes_drug = codecs.open(vocab_path)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
sub_csv = pd.read_csv('./ESPF/subword_units_map_chembl.csv')

idx2word_d = sub_csv['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

amino_char = ['?', 'A', 'V', 'L', 'I', 'F', 'W', 'M', 'P', 'G', 'S', 'T', 'C', 'Y',
              'N', 'Q', 'H', 'K', 'R', 'D', 'E']
enc_protein = OneHotEncoder().fit(np.array(amino_char).reshape(-1, 1))

smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
               '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
               'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
               'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
               'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']
enc_drug = OneHotEncoder().fit(np.array(smiles_char).reshape(-1, 1))

config = BIN_config_DBPE()


def protein_2_embed(x):
    # Direct encoding
    MAX_SEQ_PROTEIN = config['MAX_SEQ_PROTEIN']
    temp = list(x.upper())
    temp = [i if i in amino_char else '?' for i in temp]
    if len(temp) < MAX_SEQ_PROTEIN:
        temp = temp + ['?'] * (MAX_SEQ_PROTEIN - len(temp))
    else:
        temp = temp[:MAX_SEQ_PROTEIN]

    return enc_protein.transform(np.array(temp).reshape(-1, 1)).toarray().T


def drug_2_embed(x):
    # Direct encoding
    MAX_SEQ_DRUG = config['MAX_SEQ_DRUG']
    temp = list(x)
    temp = [i if i in smiles_char else '?' for i in temp]
    if len(temp) < MAX_SEQ_DRUG:
        temp = temp + ['?'] * (MAX_SEQ_DRUG - len(temp))
    else:
        temp = temp[:MAX_SEQ_DRUG]

    return enc_drug.transform(np.array(temp).reshape(-1, 1)).toarray().T


def protein2emb_encoder(x):
    max_p = 545
    t1 = pbpe.process_line(x).split()  # split
    # print(t1)
    try:
        i1 = np.asarray([words2idx_p[i] for i in t1])  # index
    except:
        i1 = np.array([0])
        # print(x)

    l = len(i1)

    if l < max_p:
        i = np.pad(i1, (0, max_p - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_p - l))
    else:
        i = i1[:max_p]
        input_mask = [1] * max_p

    return t1, i, np.asarray(input_mask)


def drug2emb_encoder(x):
    max_d = 50
    # max_d = 100
    t1 = dbpe.process_line(x).split()  # split
    # print(t1)
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])
        # print(x)

    l = len(i1)

    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))

    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return t1, i, np.asarray(input_mask)


def fp_dim_reduce(fp, max_drug_fp):
    try:
        i1 = np.where(fp == 1)[0]  # index
    except:
        i1 = np.array([0])
        # print(x)

    l = len(i1)

    if l < max_drug_fp:
        i = np.pad(i1, (0, max_drug_fp - l), 'constant', constant_values=0)
        # print(i)
        input_mask = ([1] * l) + ([0] * (max_drug_fp - l))
    else:
        i = i1[:max_drug_fp]
        input_mask = [1] * max_drug_fp

    return i, np.asarray(input_mask)


def seq2code(d, p):
    # Direct encoding
    d_v = drug_2_embed(d)
    p_v = protein_2_embed(p)
    return d_v, p_v


class BIN_Data_Encoder(data.Dataset):

    def __init__(self, list_IDs, labels, df_dti):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        d_v = {}
        p_v = {}

        index = self.list_IDs[index]
        d = self.df.iloc[index]['SMILES']
        p = self.df.iloc[index]['Target Sequence']

        if config['type'] == '1':
            # Direct encoding
            d_v['d'], p_v['p'] = seq2code(d, p)
        elif config['type'] == '2':
            # Direct encoding + Molecular fingerprint
            fp = config['fp_type']
            if fp == 'morgan':
                d_fp = np.array(json.loads(self.df.iloc[index]['morgan']))
            elif fp == 'pubchem':
                d_fp = np.array(json.loads(self.df.iloc[index]['pubchem']))

            d_v['d'], p_v['p'] = seq2code(d, p)
            d_v['fp'] = d_fp
        elif config['type'] == '3':
            # Subsequence coding
            # print(self.df.iloc[index]['Gene'])
            # print(self.df.iloc[index]['DrugBank ID'])
            d_t1, d_v['d'], d_v['input_mask_d'] = drug2emb_encoder(d)
            p_t1, p_v['p'], p_v['input_mask_p'] = protein2emb_encoder(p)
        elif config['type'] == '4':
            # Subsequence coding + Molecular fingerprint
            fp = config['fp_type']
            if fp == 'morgan':
                d_fp = np.array(json.loads(self.df.iloc[index]['morgan']))
                d_v['d_fp'], d_v['d_fp_mask'] = fp_dim_reduce(d_fp, config['max_drug_fp'])
            elif fp == 'pubchem':
                d_fp = np.array(json.loads(self.df.iloc[index]['pubchem']))
                d_v['d_fp'], d_v['d_fp_mask'] = fp_dim_reduce(d_fp, config['max_drug_fp'])

            d_v['d'], d_v['input_mask_d'] = drug2emb_encoder(d)
            p_v['p'], p_v['input_mask_p'] = protein2emb_encoder(p)

        y = self.labels[index]

        # Generate interaction diagram related files
        # fp = open('./chord/protein.txt', 'a')
        # fp.write(",".join(p_t1) + '\n')
        # fp.close()
        #
        # dp = open('./chord/drug.txt', 'a')
        # dp.write(",".join(d_t1) + '\n')
        # dp.close()

        return d_v, p_v, y
