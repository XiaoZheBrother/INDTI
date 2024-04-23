import numpy as np
import pandas as pd
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc, matthews_corrcoef
from models import BIN_Interaction_Flat
from config import BIN_config_DBPE
from stream import BIN_Data_Encoder
from torch.utils import data
import torch


def model_pretrained(path_dir=None):
    model = BIN_Interaction_Flat(**config)

    model_dict = torch.load(path_dir).module.state_dict()

    model.load_state_dict(model_dict)
    # model = model.cuda()

    return model


def test_(data_generator, model):
    y_pred = []
    y_label = []

    model.eval()

    for i, (d, p, label) in enumerate(data_generator):

        test_data = {}
        test_data['d'] = d['d'].long().cpu()
        test_data['p'] = p['p'].long().cpu()

        if config['type'] == '2':
            test_data['fp'] = d['fp'].long()
        elif config['type'] == '3':
            # Subsequence coding
            test_data['input_mask_d'] = d['input_mask_d'].long()
            test_data['input_mask_p'] = p['input_mask_p'].long()
        elif config['type'] == '4':
            # Subsequence coding + Molecular fingerprint
            test_data['d_fp'] = d['d_fp'].long().cpu()
            test_data['d_fp_mask'] = d['d_fp_mask'].long().cpu()
            test_data['input_mask_d'] = d['input_mask_d'].long().cpu()
            test_data['input_mask_p'] = p['input_mask_p'].long().cpu()

        score = model(**test_data)

        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score)).detach().cpu().numpy()

        label_ids = label.to('cpu').numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()
        break

    model.train()

    return y_pred


def test(data_generator, model):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0

    for i, (d, p, label) in enumerate(data_generator):
        test_data = {}
        test_data['d'] = d['d'].long()
        test_data['p'] = p['p'].long()

        if config['type'] == '2':
            test_data['fp'] = d['fp'].long().cuda()
        elif config['type'] == '3':
            # Subsequence coding
            test_data['input_mask_d'] = d['input_mask_d'].long()
            test_data['input_mask_p'] = p['input_mask_p'].long()
        elif config['type'] == '4':
            # Subsequence coding + Molecular fingerprint
            test_data['d_fp'] = d['d_fp'].long().cuda()
            test_data['d_fp_mask'] = d['d_fp_mask'].long().cuda()
            test_data['input_mask_d'] = d['input_mask_d'].long().cuda()
            test_data['input_mask_p'] = p['input_mask_p'].long().cuda()

        score = model(**test_data)

        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score))
        loss_fct = torch.nn.BCELoss()

        label = Variable(torch.from_numpy(np.array(label)).float())

        loss = loss_fct(logits, label)

        loss_accumulate += loss
        count += 1

        logits = logits.detach().cpu().numpy()

        label_ids = label.to('cpu').numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()

    loss = loss_accumulate / count

    y_pred_s = [1 if i >= 0.5 else 0 for i in y_pred]

    cm1 = confusion_matrix(y_label, y_pred_s)
    print('Confusion Matrix : \n', cm1)
    print('Recall : ', recall_score(y_label, y_pred_s))
    print('Precision : ', precision_score(y_label, y_pred_s))

    total1 = sum(sum(cm1))
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    print('Accuracy : ', accuracy1)

    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    print('Sensitivity : ', sensitivity1)

    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    print('Specificity : ', specificity1)

    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])

    MCC = matthews_corrcoef(y_label, outputs)
    print('MCC: ', MCC)

    return outputs, roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label,
                                                                                                       outputs), y_pred, loss.item()


if __name__ == '__main__':
    config = BIN_config_DBPE()
    # load model
    model_path = './model/type3/data1_final.pth'
    print('Using pretrained model and making predictions...')
    model = model_pretrained(model_path)

    params = {'batch_size': config['batch_size'],
              'shuffle': False,
              'num_workers': 0,
              'drop_last': False}

    # load test file
    df_test = pd.read_csv('./chord_data.csv')

    testing_set = BIN_Data_Encoder(df_test.index.values, df_test.Label.values, df_test)
    testing_generator = data.DataLoader(testing_set, **params)

    print('--- Go for Testing ---')
    try:
        with torch.set_grad_enabled(False):
            # auc, auprc, f1, logits, loss = test(testing_generator, model)
            score = test_(testing_generator, model)
            # print(
            #     'Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(f1) + ' , Test loss: ' + str(
            #         loss))
    except RuntimeError as exception:
        print('testing failed')
        print(exception)

    score = [1 if i >= 0.5 else 0 for i in score]

    df_test['pred'] = score
    df_test.to_csv('./chord_data.csv')
