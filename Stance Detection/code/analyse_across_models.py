import sys
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

from data import load_xstance, pd_read_jsonl


if __name__ == "__main__":
    labels2xstance = {"Zustimmung":"FAVOR", "Ablehnung":"AGAINST"}

    gold_path = sys.argv[1]
    pred_paths = sys.argv[2:]

    gold_data = load_xstance(gold_path).set_index("id")
    print(gold_data)

    full_data = gold_data.copy()
    for p in pred_paths:
        name = "_".join(p.split("/")[-2].split("_")[1:])+"_"+p.split("/")[-1].split(".")[1]
        pdf = pd_read_jsonl(p).set_index("id")[["pred_label"]]
        pdf["pred_label"] = pdf["pred_label"].apply(lambda x:labels2xstance[x] if x in labels2xstance else x)
        match_data = gold_data.join(pdf)
        match_data["match"] = match_data["label"]==match_data["pred_label"]
        match_data = match_data[["match"]].rename(columns={"match":"match_"+name})
        full_data = full_data.join(match_data)
    print()

    full_data["matches"] = [[v for k,v in r.items() if "match" in k] for _,r in full_data.iterrows()]
    full_data["rating"] = full_data["matches"].apply(lambda x:Counter(x)[True])


    eazy_cases = full_data[full_data["matches"].apply(lambda x:all(x))].reset_index()
    hard_cases = full_data[full_data["matches"].apply(lambda x:not any(x))].reset_index()
    middle_cases = full_data[full_data["matches"].apply(lambda x:any(x) and (not all(x)))].reset_index()
    print(len(eazy_cases))
    print(len(middle_cases))
    print(len(hard_cases))

    eazy_cases["case"] = "eazy"
    middle_cases["case"] = "middle"
    hard_cases["case"] = "hard"

    labeled = pd.concat([eazy_cases,middle_cases,hard_cases])

    labeled["AnnotatorConf"] = labeled["numerical_label"].apply(lambda x:"100%" if x in [0, 100] else "50%")

    labeled.hist(column="AnnotatorConf", by="rating")
    plt.show()

    print(labeled.groupby(["case", "AnnotatorConf"]).count().unstack())
    labeled.groupby(["case", "AnnotatorConf"]).count().unstack()["id"].plot.bar(stacked=True, rot=0)
    plt.show()

    print(labeled.groupby(["topic", "rating"]).count().unstack())
    labeled.groupby(["topic", "rating"]).count().unstack()["id"].plot.bar(stacked=True)
    plt.show()
    


    labeled.to_excel("../cross_model_anaylse.xlsx")


