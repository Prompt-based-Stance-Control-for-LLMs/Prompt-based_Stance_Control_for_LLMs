import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("tkagg")
import re 

from data import load_xstance, pd_read_jsonl



if __name__ == "__main__":
    print("Full Data(with Fr,It): ", 
    len(pd.concat([pd_read_jsonl("../data/xstance/train.jsonl"), pd_read_jsonl("../data/xstance/valid.jsonl"), pd_read_jsonl("../data/xstance/test.jsonl")])))
    train = load_xstance("../data/xstance/train.jsonl")
    valid = load_xstance("../data/xstance/valid.jsonl")
    test  = load_xstance("../data/xstance/test.jsonl")

    full = pd.concat([train,valid,test])

    print("#Instances: {} / {} / {}".format(len(train), len(valid), len(test)))

    print("FAVOR/AGAINST: {} / {}".format(len(full[full["label"]=="FAVOR"]),
                                          len(full[full["label"]=="AGAINST"])))

    print("#Topics:", len(set(full["topic"].unique())))

    question_length = full["question"].apply(lambda x:len(re.sub(r"\s", " ",x).split(" ")))
    text_length = full["comment"].apply(lambda x:len(re.sub(r"\s", " ",x).split(" ")))
    print("Avg.#Words (question): ", question_length.min(), question_length.mean(), question_length.max())
    print("Avg.#Words (comment): ", text_length.min(), text_length.mean(), text_length.max())

    print("#Questions:", len(set(full["question_id"].unique())))

    print("Examples per Topic:")
    print(full.groupby("topic").size(),"\n")

    full.groupby("topic").size().plot(kind="bar")
    plt.show()

    full.groupby("question_id").size().plot(kind="bar")
    plt.show()


    