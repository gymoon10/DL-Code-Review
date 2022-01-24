# 7장 자연어 처리에 의한 감정 분석(Transformer)

import glob
import os
import io
import string
import re
import random
import spacy
import torchtext
from torchtext.vocab import Vectors


def get_IMDb_DataLoaders_and_TEXT(max_length=256, batch_size=24):
    """IMDb의 DataLoader와 TEXT 오브젝트를 취득"""

    # 훈련 데이터의 tsv 파일을 작성합니다
    f = open('./data/IMDb_train.tsv', 'w')

    path = './data/aclImdb/train/pos/'
    for fname in glob.glob(os.path.join(path, '*.txt')):
        with io.open(fname, 'r', encoding="utf-8") as ff:
            text = ff.readline()

            # 탭이 있으면 지웁니다
            text = text.replace('\t', " ")

            text = text+'\t'+'1'+'\t'+'\n'
            f.write(text)

    path = './data/aclImdb/train/neg/'
    for fname in glob.glob(os.path.join(path, '*.txt')):
        with io.open(fname, 'r', encoding="utf-8") as ff:
            text = ff.readline()

            # 탭이 있으면 지웁니다
            text = text.replace('\t', " ")

            text = text+'\t'+'0'+'\t'+'\n'
            f.write(text)

    f.close()

   # 테스트 데이터 작성
    f = open('./data/IMDb_test.tsv', 'w')

    path = './data/aclImdb/test/pos/'
    for fname in glob.glob(os.path.join(path, '*.txt')):
        with io.open(fname, 'r', encoding="utf-8") as ff:
            text = ff.readline()

            # 탭이 있으면 지웁니다
            text = text.replace('\t', " ")

            text = text+'\t'+'1'+'\t'+'\n'
            f.write(text)

    path = './data/aclImdb/test/neg/'
    for fname in glob.glob(os.path.join(path, '*.txt')):
        with io.open(fname, 'r', encoding="utf-8") as ff:
            text = ff.readline()

            # 탭이 있으면 지웁니다
            text = text.replace('\t', " ")

            text = text+'\t'+'0'+'\t'+'\n'
            f.write(text)
    f.close()

    def preprocessing_text(text):
        # 개행 코드 삭제
        text = re.sub('<br />', '', text)

        # 쉼표, 마침표 이외의 기호를 공백으로 치환
        for p in string.punctuation:
            if (p == ".") or (p == ","):
                continue
            else:
                text = text.replace(p, " ")

        # 쉼표, 마침표의 전후에 공백 추가
        text = text.replace(".", " . ")
        text = text.replace(",", " , ")
        return text

    # 띄어쓰기(이번에는 영어 데이터이며, 임시로 공백으로 구분)
    def tokenizer_punctuation(text):
        return text.strip().split()


    # 전처리 및 띄어쓰기를 포함한 함수 정의
    def tokenizer_with_preprocessing(text):
        text = preprocessing_text(text)
        ret = tokenizer_punctuation(text)
        return ret


    # 데이터를 읽었을 때, 내용에 대해 수행할 처리를 정의합니다
    # max_length
    TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True,
                                lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

    # "data" 폴더에서 각 tsv 파일을 읽습니다
    train_val_ds, test_ds = torchtext.data.TabularDataset.splits(
        path='./data/', train='IMDb_train.tsv',
        test='IMDb_test.tsv', format='tsv',
        fields=[('Text', TEXT), ('Label', LABEL)])

    # torchtext.data.Dataset의 split 함수로 훈련 데이터와 validation 데이터를 나누기
    train_ds, val_ds = train_val_ds.split(
        split_ratio=0.8, random_state=random.seed(1234))

    # torchtext에서 단어 벡터로서 학습된 모델(영어)를 읽습니다
    english_fasttext_vectors = Vectors(name='data/wiki-news-300d-1M.vec')

    # 벡터화 된 버전의 vocabulary를 만듭니다
    TEXT.build_vocab(train_ds, vectors=english_fasttext_vectors, min_freq=10)

    # DataLoader를 작성합니다(torchtext에서는 단순히 iterater로 불립니다)
    train_dl = torchtext.data.Iterator(
        train_ds, batch_size=batch_size, train=True)

    val_dl = torchtext.data.Iterator(
        val_ds, batch_size=batch_size, train=False, sort=False)

    test_dl = torchtext.data.Iterator(
        test_ds, batch_size=batch_size, train=False, sort=False)

    return train_dl, val_dl, test_dl, TEXT
