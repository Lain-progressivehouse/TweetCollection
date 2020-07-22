import pandas as pd
import json
from collections import defaultdict
from tqdm import tqdm
import os


def read_json_lines(path: str, output="./data/tweets.csv") -> pd.DataFrame:
    """
    json形式からcsvへの変換(tweets.json専用)
    :param path: jsonのpath
    :param output: csvの出力先
    :return: pandas
    """
    json_dict = defaultdict(list)

    with open(path, "r") as fin:
        for line in tqdm(fin):
            data = json.loads(line)
            for key in data.keys():
                json_dict[key].append(list(data[key].values())[0] if isinstance(data[key], dict) else data[key])

    df = pd.DataFrame()
    for key in json_dict.keys():
        df[key] = json_dict[key]

    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_csv(output, index=False)
    return df
