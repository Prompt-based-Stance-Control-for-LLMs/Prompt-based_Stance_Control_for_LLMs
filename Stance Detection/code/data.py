import pandas as pd
import json


def pd_read_jsonl(path):
    data = []
    with open(path, "r") as ifile:
        for line in ifile:
            if line.strip()=="":
                continue
            data.append(json.loads(line))
    return pd.DataFrame(data)

def load_xstance(path):
    data_raw = pd_read_jsonl(path)
    data_de = data_raw[data_raw["language"]=="de"]
    ##
    return data_de.reset_index(drop=True)

def write_jsonl(list_of_dicts, opath):
    with open(opath, "w") as ofile:
        for d in list_of_dicts:
            ofile.write(json.dumps(d)+"\n")


def xstance_instance2text(dataset_or_dataframe):
    return dataset_or_dataframe["question"]+" | "+dataset_or_dataframe["comment"]

