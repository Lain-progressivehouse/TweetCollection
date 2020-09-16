from transformers import BertConfig, BertJapaneseTokenizer, BertModel
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from tqdm import tqdm
from src import df_maker
from logzero import logger
import pickle
from src.text_preprocess import (Compose, ReplyRemove, ZenToHan, SpaceRemove)


class BertDataset(Dataset):
    def __init__(self, df, max_length=128, model_name="bert-base-japanese-whole-word-masking", transforms=None):
        self.max_length = max_length
        self.df = df
        self.model_name = model_name
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
        self.transforms = transforms

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]
        if self.transforms is not None:
            row["full_text"] = self.transforms(row["full_text"])
        input_ids, attention_mask = self._tokenize(row["full_text"])
        data["input_ids"] = input_ids
        data["attention_mask"] = attention_mask
        return data

    def __len__(self):
        return len(self.df)

    def _tokenize(self, text):
        id_dict = tokenizer.encode_plus(str(text),
                                        max_length=self.max_length,
                                        pad_to_max_length=True,
                                        truncation=True)
        return id_dict["input_ids"], id_dict["attention_mask"]


class BertFeatures(object):

    def __init__(self, max_length=128, batch_size=64, model_name="bert-base-japanese-whole-word-masking",
                 transforms=None):
        self.max_length = max_length
        self.batch_size = batch_size
        self.model_name = model_name
        self.config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用デバイス: {device}")
        self.bert.to(device)
        self.transforms = transforms

    def get_features(self, df):
        ds = Dataset(
            df,
            max_length=self.max_length,
            model_name=self.model_name,
            transforms=self.transforms
        )
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        preds = []
        for batch in tqdm(dl):
            input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            output = bert(input_ids=input_ids, attention_mask=attention_mask)
            output = output[0]
            output = output.to(cpu)
            preds.append(output.detach().clone().numpy())
        return np.concatenate(preds, axis=0)


# MODEL_NAME = "bert-base-japanese-whole-word-masking"
MODEL_NAME = "bert-base-japanese-whole-word-masking"
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
                                    pad_to_max_length=True,
                                    truncation=True)
    return id_dict["input_ids"], id_dict["attention_mask"]


def main():
    # df = df_maker.read_json_lines("./data/tweets")
    df = pd.read_csv("./data/tweets.csv")
    df = df.iloc[:100000]

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
