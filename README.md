# INDTI: Drug target interactions prediction based on interactive inference network

Overview of INDTI 

<sup>INDTI has an embedding layer, an encoding layer, an interaction layer, a feature extraction layer, and an output layer. At the embedding layer, drug sequences and target molecules are embedded. The embedding of drugs and targets is encoded at the encoding layer. The interaction layer simulates drug-target interactions. In the feature extraction layer, the interaction features of the interaction matrix are extracted and the prediction results are obtained. In general, when predicting the interaction between a target and a drug, most models connect the extracted features of both the drug and the target molecular sequences, feeding them into the prediction model. However, our model uses the dot product to generate a scalar indicating the interaction strength between a single target-drug minimum unit pair, which produces interpretable model interaction predictions.<sup>

![INDTI](https://github.com/XiaoZheBrother/INDTI/blob/main/INDTI.jpg "INDTI")

## Files
*train.py*: Training function of INDTI:\
Set the data set and train the model\
First, modify the corresponding configuration in the config.py file. Then select the dataset for training\
*test.py*: Test function of INDTI:\
Specify the data to be used for prediction, make model prediction, and generate interactive files at the same time\
First, modify the corresponding configuration in the config.py file. Then select the test set, and refer to val.csv for the format of the test set\
*stream.py*: Used to generate embedded\
*models.py*: INDTI model\
*config.py*: The config of INDTI:\
Type1 represents Direct embedding + CNN\
Type2 represents Direct embedding + Molecular fingerprint + CNN\
Type3 represents Subsequent embedding + Attention\
Type4 represents Subsequent embedding + Molecular fingerprint + Attention\
fp_type is used to specify PubChem or Morgan\
*CNN.py*: The CNN encoder\
*self_attention.py*: The Self-Attention encoder\
*chord*: Example of interactive file\
*data*: Example of preprocessing data\
*dataset*: Data from three datasets:
BindingDB,BIOSNAP and DAVIS\
*ESPF*: Generate embedded related files\
*model*: Example of Trained model(https://stumail-my.sharepoint.cn/:u:/g/personal/21yqchen2_stu_edu_cn/Edr1wyTOvAFEnLRpf6Sjqr4BKxqUr0KoAgD67lYuCc27jA)
