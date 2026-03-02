import sys
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

from data import load_xstance, pd_read_jsonl


if __name__ == "__main__":
    gold_path = "data/xstance/valid.jsonl"
    
    english_predictions = [
        "experiment_1/exp1_english_example/val.gemma.jsonl",
        "experiment_1/exp1_english_example/val.qwen14b.jsonl",
        "experiment_1/exp1_german_bert/6348.valid.predictions.jsonl",
    ]
    german_predictions = [
        "experiment_1/exp1_german_example/valid.qwen_72b.jsonl",
        "experiment_1/exp1_german_simple/val.sauerkraut.jsonl",
        "experiment_1/exp1_german_example/val.gemma.jsonl",
        "experiment_1/exp1_german_example/val.qwen14b.jsonl",
    ]

    labels2xstance = {"Zustimmung":"FAVOR", "Ablehnung":"AGAINST"}


    gold_data = load_xstance(gold_path).set_index("id")
    print(gold_data)

    model1 = "experiment_1/exp1_german_example/valid.qwen_72b.jsonl"
    model2 = "experiment_1/exp1_german_example/val.qwen14b.jsonl"
    preds1 = pd_read_jsonl(model1).set_index("id")[["pred_label"]].rename(columns={"pred_label":"model1"})
    preds2 = pd_read_jsonl(model2).set_index("id")[["pred_label"]].rename(columns={"pred_label":"model2"})

    data_full = gold_data.join(preds1).join(preds2)

    data_full["model1_match"] = data_full["label"]==data_full["model1"].apply(lambda x:labels2xstance[x])
    data_full["model2_match"] = data_full["label"]==data_full["model2"].apply(lambda x:labels2xstance[x])
    data_full["model_aggree"] = data_full["model1"]==data_full["model2"]

    data_full.to_excel("model_analysis_stats.xlsx")