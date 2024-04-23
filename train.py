import copy
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc, matthews_corrcoef
from torch import nn
from torch.autograd import Variable
from torch.utils import data

torch.manual_seed(2)
np.random.seed(3)

from config import BIN_config_DBPE
from models import BIN_Interaction_Flat
from stream import BIN_Data_Encoder

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def test(data_generator, model):
    config = BIN_config_DBPE()

    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for i, (d, p, label) in enumerate(data_generator):
        test_data = {}
        test_data['d'] = d['d'].long().to(device)
        test_data['p'] = p['p'].long().to(device)

        if config['type'] == '2':
            test_data['fp'] = d['fp'].long().to(device)
        elif config['type'] == '3':
            # Subsequence coding
            test_data['input_mask_d'] = d['input_mask_d'].long().to(device)
            test_data['input_mask_p'] = p['input_mask_p'].long().to(device)
        elif config['type'] == '4':
            # Subsequence coding + Molecular fingerprint
            test_data['input_mask_d'] = d['input_mask_d'].long().to(device)
            test_data['input_mask_p'] = p['input_mask_p'].long().to(device)
            test_data['d_fp'] = d['d_fp'].long().to(device)
            test_data['d_fp_mask'] = d['d_fp_mask'].long().to(device)

        score = model(**test_data)

        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score))
        loss_fct = torch.nn.BCELoss()

        label = Variable(torch.from_numpy(np.array(label)).float()).to(device)

        loss = loss_fct(logits, label)

        loss_accumulate += loss
        count += 1

        logits = logits.detach().cpu().numpy()

        label_ids = label.to('cpu').numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()

    loss = loss_accumulate / count

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)

    y_pred_s = [1 if i >= 0.5 else 0 for i in y_pred]

    auc_k = auc(fpr, tpr)
    print("AUROC:" + str(auc_k))
    print("AUPRC: " + str(average_precision_score(y_label, y_pred)))

    cm1 = confusion_matrix(y_label, y_pred_s)
    print('Confusion Matrix : \n', cm1)
    print('Recall : ', recall_score(y_label, y_pred_s))
    print('Precision : ', precision_score(y_label, y_pred_s))

    total1 = sum(sum(cm1))
    # from confusion matrix calculate accuracy
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    print('Accuracy : ', accuracy1)

    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    print('Sensitivity : ', sensitivity1)

    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    print('Specificity : ', specificity1)

    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])

    MCC = matthews_corrcoef(y_label, outputs)
    print('MCC: ', MCC)

    return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label,
                                                                                              outputs), y_pred, loss.item()


def main():
    config = BIN_config_DBPE()
    print(config)

    for folder in ['1']:
        model = BIN_Interaction_Flat(**config)

        model = model.to(device)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model, dim=0)

        opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
        print('--- Data Preparation ---')
        params = {'batch_size': config['batch_size'],
                  'shuffle': True,
                  'num_workers': 0,
                  'drop_last': True}
        loss_history = []

        dataFolder = './data/data{}'.format(folder)

        df_train = pd.read_csv(dataFolder + '/train.csv')
        df_val = pd.read_csv(dataFolder + '/val.csv')
        df_test = pd.read_csv(dataFolder + '/test.csv')

        training_set = BIN_Data_Encoder(df_train.index.values, df_train.Label.values, df_train)
        training_generator = data.DataLoader(training_set, **params)

        validation_set = BIN_Data_Encoder(df_val.index.values, df_val.Label.values, df_val)
        validation_generator = data.DataLoader(validation_set, **params)

        testing_set = BIN_Data_Encoder(df_test.index.values, df_test.Label.values, df_test)
        testing_generator = data.DataLoader(testing_set, **params)

        # early stopping
        model_max = copy.deepcopy(model)
        with torch.set_grad_enabled(False):
            auc, auprc, f1, logits, loss = test(testing_generator, model_max)
            print('Initial Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(
                f1) + ' , Test loss: ' + str(loss))

        print('--- Go for Training ---')
        torch.backends.cudnn.benchmark = True

        start_epoch = 0

        for epo in range(start_epoch, config['epochs']):
            model.train()
            for i, (d, p, label) in enumerate(training_generator):
                train_data = {}
                train_data['d'] = d['d'].long().to(device)
                train_data['p'] = p['p'].long().to(device)

                if config['type'] == '2':
                    train_data['fp'] = d['fp'].long().to(device)

                elif config['type'] == '3':
                    # Subsequence coding
                    train_data['input_mask_d'] = d['input_mask_d'].long().to(device)
                    train_data['input_mask_p'] = p['input_mask_p'].long().to(device)
                elif config['type'] == '4':
                    # Subsequence coding + Molecular fingerprint
                    train_data['input_mask_d'] = d['input_mask_d'].long().to(device)
                    train_data['input_mask_p'] = p['input_mask_p'].long().to(device)
                    train_data['d_fp'] = d['d_fp'].long().to(device)
                    train_data['d_fp_mask'] = d['d_fp_mask'].long().to(device)

                score = model(**train_data)

                label = Variable(torch.from_numpy(np.array(label)).float()).to(device)

                loss_fct = torch.nn.BCELoss()
                m = torch.nn.Sigmoid()
                n = torch.squeeze(m(score))

                loss = loss_fct(n, label)
                loss_history.append(loss)

                opt.zero_grad()
                loss.backward()
                opt.step()

                if i % 100 == 0:
                    print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' with loss ' + str(
                        loss.cpu().detach().numpy()))

            # every epoch test
            with torch.set_grad_enabled(False):
                auc, auprc, f1, logits, loss = test(validation_generator, model)
                print('Validation at Epoch ' + str(epo + 1) + ' , AUROC: ' + str(auc) + ' , AUPRC: ' + str(
                    auprc) + ' , F1: ' + str(f1) + ' , loss: ' + str(loss))

        print('--- Go for Testing ---')
        model_max = copy.deepcopy(model)
        try:
            with torch.set_grad_enabled(False):
                auc, auprc, f1, logits, loss = test(testing_generator, model_max)
                print('Test{}:'.format(folder))
                print(
                    'Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(
                        f1) + ' , Test loss: ' + str(
                        loss))
        except RuntimeError as exception:
            print('testing failed')


print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
main()
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
