import gensim.models
from gensim.test.utils import datapath
import os
import numpy as np
import joblib
import pandas as pd

class Swem(object):
    """
    Swemを取得するクラス
    """

    def __init__(self, file_path: str = None, is_train: bool = True):
        """
        file_pathにchiveのpathを指定する
        指定しない場合，新たに学習する
        is_trainがTrueの場合，追加学習を行う
        :param file_path: chiveのpath
        :param is_train: 追加学習を行うかどうか
        """
        self.file_path = file_path
        self.is_train = is_train
        self.model: gensim.models.Word2Vec = \
            gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=False) if file_path else None
        self.vocab = set(self.model.vocab.keys()) if file_path else None
        self.vector_size = self.model.vector_size if file_path else None

    def _train(self, sentences) -> None:
        """
        追加学習を行う
        :param sentences: 単語に分割されたリスト
        """
        self.model.build_vocab(sentences, update=True)
        self.model.train(sentences, total_examples=self.model.corpus_count, epochs=self.model.iter)
        self.vocab = set(self.model.vocab.keys())

    def save_model(self, file_path: str) -> None:
        """
        モデルを保存する
        :param file_path: 保存先
        """
        assert file_path, "file_path is None"

        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        # self.model.save(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self.model, file_path, compress=True)

    def load_model(self, file_path: str) -> None:
        """
        モデルのロード
        :param file_path:
        :return:
        """
        assert os.path.exists(file_path), "file_path is not exists"
        # self.model = gensim.models.Word2Vec.load(file_path)
        self.model = joblib.load(file_path)
        self.vocab = set(self.model.vocab.keys())
        self.vector_size = self.model.vector_size

    def _get_word_embeddings(self, word_list) -> np.array:
        """
        word_embeddingのリストを取得する
        :param word_list: 単語リスト
        :return: np.array, shape(len(word_list), self.vector_size)
        """
        np.random.seed(abs(hash(" ".join(word_list))) % (10 ** 8))
        vectors = []
        for word in word_list:
            if word in self.vocab:
                vectors.append(self.model[word])
            else:
                vectors.append(np.random.uniform(-0.01, 0.01, self.vector_size))

        return np.array(vectors)

    def get_swem(self, sentences) -> np.array:
        """
        swemの構築
        :param sentences: 単語に分割されたリスト
        :return:
        """
        if self.file_path and self.is_train:
            self._train(sentences)

        swems = []
        for word_list in sentences:
            word_embeddings = self._get_word_embeddings(word_list)
            swems.append(np.r_[np.mean(word_embeddings, axis=0), np.max(word_embeddings, axis=0)])

        return np.array(swems)

def main():
    swem = Swem(file_path="./data/model/chive-1.1-mc5-20200318/chive-1.1-mc5-20200318.txt", is_train=True)
    df = pd.read_csv("./data/tweets_wakati_hinshi.csv")

