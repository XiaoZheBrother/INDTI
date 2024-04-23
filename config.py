def BIN_config_DBPE():
    config = {}
    config['batch_size'] = 8
    config['epochs'] = 50
    config['lr'] = 5e-4
    config['type'] = '3'  # 1:Direct encoding + CNN/2:Direct encoding + Molecular fingerprint + CNN/3:Subsequence
    # coding + attention/4:Subsequence coding + Molecular fingerprint + attention
    config['fp_type'] = 'morgan'

    # CNN
    config['cnn_target_filters'] = [32, 64, 96]
    config['cnn_target_kernels'] = [4, 8, 12]
    config['cnn_drug_filters'] = [32, 64, 96]
    config['cnn_drug_kernels'] = [4, 8, 12]
    config['hidden_dim_drug'] = 256
    config['hidden_dim_protein'] = 256
    config['MAX_SEQ_PROTEIN'] = 1000
    config['MAX_SEQ_DRUG'] = 100
    config['cnn_hidden_dims'] = [256, 128, 32]

    # MLP
    # if puchem，num_input is 881，if morgan，num_input is 1024
    num_input = 881
    config['mlp_input_dim'] = num_input
    config['mlp_hidden_dim'] = 256
    config['mlp_hidden_dims'] = [num_input, 256, 64]

    # fp
    config['input_dim_fp'] = 1024
    config['max_drug_fp'] = 76
    config['output_dim_fp'] = 126

    config['input_dim_drug'] = 23532
    config['input_dim_target'] = 16693
    config['train_epoch'] = 13
    config['max_drug_seq'] = 50
    config['max_protein_seq'] = 545
    config['emb_size'] = 384
    config['dropout_rate'] = 0.1

    # DenseNet
    config['scale_down_ratio'] = 0.25
    config['growth_rate'] = 20
    config['transition_rate'] = 0.5
    config['num_dense_blocks'] = 4
    config['kernal_dense_size'] = 3

    # Encoder
    config['intermediate_size'] = 1536
    config['num_attention_heads'] = 12
    config['attention_probs_dropout_prob'] = 0.1
    config['hidden_dropout_prob'] = 0.1
    config['flat_dim'] = 78192
    return config
