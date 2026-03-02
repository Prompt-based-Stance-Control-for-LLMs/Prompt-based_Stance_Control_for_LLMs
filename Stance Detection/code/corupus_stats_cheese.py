import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("tkagg")
import re 

from data import load_xstance, pd_read_jsonl



if __name__ == "__main__":
    full = pd_read_jsonl("../data/cheese/cheese.expanded.jsonl")
    print("Full Data(: ", len(full))


    #print("#Instances: {} / {} / {}".format(len(train), len(valid), len(test)))

    print("Zustimmung/Neutral/Ablehnung: {} / {} / {}".format(
        len(full[full["label"]=="Zustimmung"]),
        len(full[full["label"]=="Neutral"]),
        len(full[full["label"]=="Ablehnung"])))

    print("#Topics:", len(set(full["topic"].unique())))

    question_length = full["question"].apply(lambda x:len(re.sub(r"\s", " ",x).split(" ")))
    text_length = full["comment"].apply(lambda x:len(re.sub(r"\s", " ",x).split(" ")))
    print("Avg.#Words (question): ", question_length.min(), question_length.mean(), question_length.max())
    print("Avg.#Words (comment): ", text_length.min(), text_length.mean(), text_length.max())

    print("Examples per Topic:")
    print(full.groupby("topic").size(),"\n")

    full.groupby("topic").size().plot(kind="bar")
    plt.show()

    