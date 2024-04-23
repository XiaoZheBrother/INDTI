from __future__ import print_function
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from helper import LayerNorm
from self_attention import Encoder_MultipleLayers
from CNN import CNN, MLP

torch.manual_seed(1)
np.random.seed(1)


class BIN_Interaction_Flat(nn.Sequential):
    """
        Interaction Network with 2D interaction map
    """

    def __init__(self, **config):
        super(BIN_Interaction_Flat, self).__init__()

        self.type = config['type']

        # CNN
        self.cnn_target_filters = config['cnn_target_filters']
        self.cnn_target_kernels = config['cnn_target_kernels']
        self.cnn_drug_filters = config['cnn_drug_filters']
        self.cnn_drug_kernels = config['cnn_drug_kernels']
        self.hidden_dim_drug = config['hidden_dim_drug']
        self.hidden_dim_protein = config['hidden_dim_protein']

        # mlp
        self.mlp_input_dim = config['mlp_input_dim']
        self.mlp_hidden_dim = config['mlp_hidden_dim']
        self.mlp_hidden_dims = config['mlp_hidden_dims']

        self.max_d = config['max_drug_seq']
        self.max_p = config['max_protein_seq']
        self.emb_size = config['emb_size']
        self.dropout_rate = config['dropout_rate']
        self.cnn_hidden_dims = config['cnn_hidden_dims']

        # densenet
        self.scale_down_ratio = config['scale_down_ratio']
        self.growth_rate = config['growth_rate']
        self.transition_rate = config['transition_rate']
        self.num_dense_blocks = config['num_dense_blocks']
        self.kernal_dense_size = config['kernal_dense_size']
        self.batch_size = config['batch_size']
        self.input_dim_drug = config['input_dim_drug']
        self.input_dim_target = config['input_dim_target']
        self.gpus = torch.cuda.device_count()
        self.n_layer = 2
        # encoder
        self.hidden_size = config['emb_size']
        self.intermediate_size = config['intermediate_size']
        self.num_attention_heads = config['num_attention_heads']
        self.attention_probs_dropout_prob = config['attention_probs_dropout_prob']
        self.hidden_dropout_prob = config['hidden_dropout_prob']

        self.flatten_dim = config['flat_dim']

        # fp
        self.input_dim_fp = config['input_dim_fp']
        self.max_drug_fp = config['max_drug_fp']
        self.output_dim_fp = config['output_dim_fp']

        # specialized embedding with positional one
        self.demb = Embeddings(self.input_dim_drug, self.emb_size, self.max_d, self.dropout_rate)
        self.pemb = Embeddings(self.input_dim_target, self.emb_size, self.max_p, self.dropout_rate)
        self.dfpemb = Embeddings(self.input_dim_fp, self.emb_size, self.max_drug_fp, self.dropout_rate)

        self.d_encoder = Encoder_MultipleLayers(self.n_layer, self.hidden_size, self.intermediate_size,
                                                self.num_attention_heads, self.attention_probs_dropout_prob,
                                                self.hidden_dropout_prob)
        self.p_encoder = Encoder_MultipleLayers(self.n_layer, self.hidden_size, self.intermediate_size,
                                                self.num_attention_heads, self.attention_probs_dropout_prob,
                                                self.hidden_dropout_prob)
        self.dfp_encoder = Encoder_MultipleLayers(self.n_layer, self.hidden_size, self.intermediate_size,
                                                  self.num_attention_heads, self.attention_probs_dropout_prob,
                                                  self.hidden_dropout_prob)

        self.d_cnn_encoder = CNN('drug', self.cnn_drug_filters, self.cnn_drug_kernels, self.hidden_dim_drug)
        self.p_cnn_encoder = CNN('protein', self.cnn_target_filters, self.cnn_target_kernels, self.hidden_dim_protein)

        self.fp_mlp_encoder = MLP(self.mlp_input_dim, self.mlp_hidden_dim, self.mlp_hidden_dims)
        self.d_fp_mlp_encoder = nn.Linear(self.mlp_hidden_dim + self.hidden_dim_drug, self.hidden_dim_protein)
        # self.d_fp_cnn_encoder = nn.Conv1d(in_channels=self.emb_size, out_channels=self.emb_size, kernel_size=4)

        self.icnn = nn.Conv2d(1, 3, 3, padding=0)

        layer_size = len(self.cnn_hidden_dims)
        dims = self.cnn_hidden_dims + [1]
        self.cnn_decoder = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])

        self.attention_decoder = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(True),

            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(True),

            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),

            # output layer
            nn.Linear(32, 1)
        )

    def forward(self, **train_data):
        d = train_data['d']
        p = train_data['p']

        if self.type == '1':
            # Direct encoding
            d_aug = self.d_cnn_encoder(d.float())
            p_aug = self.p_cnn_encoder(p.float())
            # print(d_aug.shape) torch.Size([16, 256])
            # print(p_aug.shape) torch.Size([16, 256])

            i_v = d_aug * p_aug  # interaction
            # print(i_v.shape) torch.Size([16, 256])

            for i, l in enumerate(self.cnn_decoder):
                if i == (len(self.cnn_decoder) - 1):
                    score = l(i_v)
                else:
                    f = F.dropout(i_v, p=self.dropout_rate)
                    i_v = F.relu(l(f))

        elif self.type == '2':
            d_aug = self.d_cnn_encoder(d.float())  # torch.Size([16, 128])
            p_aug = self.p_cnn_encoder(p.float())  # torch.Size([16, 256])
            # print(d_aug.shape)
            # print(p_aug.shape)

            fp = train_data['fp']
            fp_aug = self.fp_mlp_encoder(fp)  # torch.Size([16, 128])
            # print(fp_aug.shape)

            d_fp = torch.cat((d_aug, fp_aug), dim=1)  # torch.Size([16, 256])
            # print(d_aug.shape)

            d_aug = F.relu(self.d_fp_mlp_encoder(d_fp))

            i_v = d_aug * p_aug  # interaction torch.Size([16, 256])
            # print(i_v.shape)

            for i, l in enumerate(self.cnn_decoder):
                if i == (len(self.cnn_decoder) - 1):
                    score = l(i_v)
                else:
                    f = F.dropout(i_v, p=self.dropout_rate)
                    i_v = F.relu(l(f))

        elif self.type == '3':
            # Subsequence coding
            d_mask = train_data['input_mask_d']
            p_mask = train_data['input_mask_p']

            ex_d_mask = d_mask.unsqueeze(1).unsqueeze(2)
            ex_p_mask = p_mask.unsqueeze(1).unsqueeze(2)

            ex_d_mask = (1.0 - ex_d_mask) * -10000.0
            ex_p_mask = (1.0 - ex_p_mask) * -10000.0

            d_emb = self.demb(d)  # batch_size x seq_length x embed_size
            p_emb = self.pemb(p)

            d_encoded_layers = self.d_encoder(d_emb.float(), ex_d_mask.float())
            # print(d_encoded_layers.shape)
            p_encoded_layers = self.p_encoder(p_emb.float(), ex_p_mask.float())
            # print(p_encoded_layers.shape)

            # repeat to have the same tensor size for aggregation
            d_aug = torch.unsqueeze(d_encoded_layers, 2).repeat(1, 1, self.max_p, 1)  # repeat along protein size
            p_aug = torch.unsqueeze(p_encoded_layers, 1).repeat(1, self.max_d, 1, 1)  # repeat along drug size

            i = d_aug * p_aug  # interaction

            # Generate interactive files
            i_test = torch.sum(i, dim=3)
            print(i_test)
            np.save("chord/value.npy", np.array(i_test))

            i_v = i.view(int(self.batch_size / self.gpus), -1, self.max_d, self.max_p)
            i_v = torch.sum(i_v, dim=1)

            i_v = torch.unsqueeze(i_v, 1)
            # print(i_v.shape)

            i_v = F.dropout(i_v, p=self.dropout_rate)
            f = self.icnn(i_v)
            f = f.view(int(self.batch_size / self.gpus), -1)
            # print(f.shape)

            score = self.attention_decoder(f)

        elif self.type == '4':
            # Subsequence coding + Molecular fingerprint
            d_mask = train_data['input_mask_d']
            p_mask = train_data['input_mask_p']

            d_fp = train_data['d_fp']
            d_fp_mask = train_data['d_fp_mask']

            ex_d_mask = d_mask.unsqueeze(1).unsqueeze(2)
            ex_p_mask = p_mask.unsqueeze(1).unsqueeze(2)
            d_fp_mask = d_fp_mask.unsqueeze(1).unsqueeze(2)

            ex_d_mask = (1.0 - ex_d_mask) * -10000.0
            ex_p_mask = (1.0 - ex_p_mask) * -10000.0
            ex_d_fp_mask = (1.0 - d_fp_mask) * -10000.0

            d_emb = self.demb(d)  # batch_size x seq_length x embed_size
            p_emb = self.pemb(p)
            dfp_emb = self.dfpemb(d_fp)

            d_encoded_layers = self.d_encoder(d_emb.float(), ex_d_mask.float())
            # print(d_encoded_layers.shape)
            p_encoded_layers = self.p_encoder(p_emb.float(), ex_p_mask.float())
            # print(p_encoded_layers.shape)
            dfp_encoded_layers = self.dfp_encoder(dfp_emb.float(), ex_d_fp_mask.float())
            # print(dfp_encoded_layers.shape)

            d_encoded_layers = torch.cat((d_encoded_layers, dfp_encoded_layers), dim=1)

            # repeat to have the same tensor size for aggregation
            d_aug = torch.unsqueeze(d_encoded_layers, 2).repeat(1, 1, self.max_p, 1)  # repeat along protein size
            p_aug = torch.unsqueeze(p_encoded_layers, 1).repeat(1, self.output_dim_fp, 1, 1)  # repeat along drug size

            i = d_aug * p_aug  # interaction

            i_v = i.view(int(self.batch_size / self.gpus), -1, self.output_dim_fp, self.max_p)
            i_v = torch.sum(i_v, dim=1)
            # print(i_v.shape)
            i_v = torch.unsqueeze(i_v, 1)
            # print(i_v.shape)

            i_v = F.dropout(i_v, p=self.dropout_rate)
            f = self.icnn(i_v)
            f = f.view(int(self.batch_size / self.gpus), -1)
            # print(f.shape)

            score = self.attention_decoder(f)

        return score


class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """

    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
