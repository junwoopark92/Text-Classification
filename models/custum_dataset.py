import pandas as pd
import numpy as np

def testcase_shuffle_data_load():
    df = pd.read_csv('/Users/1003874/bsdev/Text-classification-with-CNN-RNN-with-Tensorflow/Ch01_Data_load/data/preprocessed_mart_digi_speech_act.csv')
    df['clean_text'] = df['clean_text'].apply(lambda x:x.lower())
    s = df['class'].value_counts()
    cols = s[s>50].index.tolist()

    sub_list = []
    for col in cols:
        sub_list.append(df[df['class'] == col])
    df = pd.concat(sub_list, axis=0)
    df = df.sample(len(df), random_state=10).reset_index(drop=True)

    added_df = pd.read_csv('/Users/1003874/bsdev/Text-classification-with-CNN-RNN-with-Tensorflow/Ch01_Data_load/data/added_dataset.csv',sep='\t')
    added_df['clean_text'] = added_df['sent'].apply(lambda x:x.lower())

    # class 제거
    added_df = added_df[added_df['class'].apply(lambda x: True if x in cols else False)]
    added_df.info()

    # duplicated sent 제거
    added_df = added_df[added_df['clean_text'].apply(lambda x: False if x in df['clean_text'].tolist() else True)]
    added_df.info()

    # train_df merge

    size = len(df)
    rate = 0.2
    train_df = df.iloc[:int(size * (1 - rate))]
    test_df = df.iloc[int(size * (1 - rate)):]

    all_df = pd.concat([train_df[['clean_text', 'class']],
                        test_df[['clean_text', 'class']],
                        added_df[['clean_text', 'class']]])

    all_df = all_df.sample(len(all_df), random_state=10)
    all_df.reset_index(drop=True, inplace=True)

    label_df = pd.get_dummies(all_df['class'])

    train_df = all_df.iloc[1062:]
    test_df = all_df.iloc[:1062]

    cols = label_df.columns

    print(len(train_df), len(test_df))

    TRAIN_DOC, TRAIN_LABEL, TEST_DOC, TEST_LABEL = \
        train_df['clean_text'].values, label_df.loc[train_df.index].values, test_df['clean_text'].values, label_df.loc[
            test_df.index].values

    return TRAIN_DOC, TRAIN_LABEL, TEST_DOC, TEST_LABEL, cols

def testcase_add_data_load():
    df = pd.read_csv('/Users/1003874/bsdev/Text-classification-with-CNN-RNN-with-Tensorflow/Ch01_Data_load/data/preprocessed_mart_digi_speech_act.csv')
    df['clean_text'] = df['clean_text'].apply(lambda x:x.lower())
    s = df['class'].value_counts()
    cols = s[s>50].index.tolist()

    sub_list = []
    for col in cols:
        sub_list.append(df[df['class'] == col])
    df = pd.concat(sub_list, axis=0)
    df = df.sample(len(df), random_state=10).reset_index(drop=True)

    added_df = pd.read_csv(
        '/Users/1003874/bsdev/Text-classification-with-CNN-RNN-with-Tensorflow/Ch01_Data_load/data/added_dataset.csv',
        sep='\t')
    added_df['clean_text'] = added_df['sent'].apply(lambda x:x.lower())

    # class 제거
    added_df = added_df[added_df['class'].apply(lambda x: True if x in cols else False)]
    added_df.info()

    # duplicated sent 제거
    added_df = added_df[added_df['clean_text'].apply(lambda x: False if x in df['clean_text'].tolist() else True)]
    added_df.info()

    # train_df merge

    size = len(df)
    rate = 0.2
    train_df = df.iloc[:int(size * (1 - rate))]
    test_df = df.iloc[int(size * (1 - rate)):]
    train_df['setlabel'] = 1
    test_df['setlabel'] = 0
    added_df['setlabel'] = 2

    all_df = pd.concat([train_df[['clean_text', 'class', 'setlabel']],
                        test_df[['clean_text', 'class', 'setlabel']],
                        added_df[['clean_text', 'class','setlabel']]])

    all_df.reset_index(drop=True, inplace=True)

    label_df = pd.get_dummies(all_df['class'])

    train_df = all_df[all_df['setlabel'] != 0]
    test_df = all_df[all_df['setlabel'] == 0]

    cols = label_df.columns

    print(len(train_df), len(test_df))

    TRAIN_DOC, TRAIN_LABEL, TEST_DOC, TEST_LABEL = \
        train_df['clean_text'].values, label_df.loc[train_df.index].values, test_df['clean_text'].values, label_df.loc[
            test_df.index].values

    return TRAIN_DOC, TRAIN_LABEL, TEST_DOC, TEST_LABEL, cols

def digi_data_load():
    df = pd.read_csv('/Users/1003874/bsdev/Text-classification-with-CNN-RNN-with-Tensorflow/Ch01_Data_load/data/preprocessed_mart_digi_speech_act.csv')
    df['clean_text'] = df['clean_text'].apply(lambda x:x.lower())
    s = df['class'].value_counts()
    cols = s[s>50].index.tolist()

    sub_list = []
    for col in cols:
        sub_list.append(df[df['class'] == col])
    df = pd.concat(sub_list, axis=0)
    df = df.sample(len(df), random_state=10).reset_index(drop=True)

    size = len(df)
    rate = 0.2
    train_df = df.iloc[:int(size*(1-rate))]
    test_df = df.iloc[int(size*(1-rate)):]
    label_df = pd.get_dummies(df['class'])
    cols = label_df.columns
    print(len(train_df), len(test_df))

    TRAIN_DOC, TRAIN_LABEL, TEST_DOC, TEST_LABEL = \
        train_df['clean_text'].values, label_df.loc[train_df.index].values, test_df['clean_text'].values, label_df.loc[test_df.index].values

    return TRAIN_DOC, TRAIN_LABEL, TEST_DOC, TEST_LABEL, cols