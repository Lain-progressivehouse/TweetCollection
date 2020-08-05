from transformers import BertConfig, BertJapaneseTokenizer, BertModel
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from src import df_maker
from logzero import logger
import pickle
from src.text_preprocess import (Compose, ReplyRemove, ZenToHan, SpaceRemove)


class BertFeatures(object):

    def __init__(self, max_length=256, batch_size=64, model_name="bert-base-japanese-whole-word-masking"):
        self.max_length = max_length
        self.batch_size = batch_size
        self.model_name = model_name
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
        self.config = BertConfig.from_pretrained(MODEL_NAME)
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用デバイス: {device}")
        self.bert.to(device)

    def _tokenize(self, text):
        id_dict = tokenizer.encode_plus(str(text),
                                        max_length=self.max_length,
                                        pad_to_max_length=True)
        return id_dict["input_ids"], id_dict["attention_mask"]

    def get_features(self, df):
        df[["TEXT", "MASK"]] = df.apply(lambda x: self._tokenize(x["full_text"]), axis=1, result_type="expand")
        ds = TensorDataset(torch.tensor(df["TEXT"], dtype=torch.int64),
                           torch.tensor(df["MASK"], dtype=torch.int64))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        preds = []
        for input_ids, attention_mask in tqdm(dl):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            output = bert(input_ids=input_ids, attention_mask=attention_mask)
            output = output[0]
            output = output.to(cpu)
            preds.append(output.detach().clone().numpy())
        return np.concatenate(preds, axis=0)


# MODEL_NAME = "bert-base-japanese-whole-word-masking"
MODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
config = BertConfig.from_pretrained(MODEL_NAME)
bert = BertModel.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device('cpu')
print(f"使用デバイス: {device}")
bert.to(device)


def tokenize(text, max_length=128):
    id_dict = tokenizer.encode_plus(str(text),
                                    max_length=max_length,
                                    pad_to_max_length=True)
    return id_dict["input_ids"], id_dict["attention_mask"]


def main():
    df = df_maker.read_json_lines("./data/tweets")

    transform = Compose(
        [
            ReplyRemove(),
            ZenToHan(),
            SpaceRemove()
        ]
    )
    df["full_text"] = df["full_text"].apply(transform)

    df[["TEXT", "MASK"]] = df.apply(lambda x: tokenize(x["full_text"]), axis=1, result_type="expand")

    bert_features = get_features(df)
    # num = bert_features.shape[1]
    # cols = [f"bert_{i}" for i in range(num)]
    # bert_features_df = pd.DataFrame(bert_features, columns=cols)
    # bert_features_df.to_csv("./data/input/bert_features.csv", index=False)
    with open("./data/input/bert_features.pkl", "wb") as f:
        pickle.dump(bert_features, f)


def get_features(df):
    ds = TensorDataset(torch.tensor(df["TEXT"], dtype=torch.int64),
                       torch.tensor(df["MASK"], dtype=torch.int64))
    dl = DataLoader(ds, batch_size=64, shuffle=False)
    preds = []
    for input_ids, attention_mask in tqdm(dl):
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        output = bert(input_ids=input_ids, attention_mask=attention_mask)
        output = output[0]
        output = output.to(cpu)
        preds.append(output.detach().clone().numpy())
    return np.concatenate(preds, axis=0)
