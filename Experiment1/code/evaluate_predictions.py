import argparse
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data import load_xstance,pd_read_jsonl


def parse_args():
    parser = argparse.ArgumentParser(prog='Evaluate predictions for XStance data.')
    parser.add_argument('labels', type=str, 
                        help='specify FAVOR,AGAINST labels sperated by a comma. Always start with the FAVOR label! Eg. "zustimmmung,ablehnung"')
    parser.add_argument('og_data', type=str, 
                        help='path to original data (eg. valid.json)')
    parser.add_argument('pred_data', type=str, 
                        help='path to predicted data')
    parser.add_argument('output_path', type=str, 
                        help='Excel (.xlsx) file where output will be stored')
    return vars(parser.parse_args())

def evaluate(df):
    A = accuracy_score(df["label"], df["pred_label"])
    P,R,F1,_ = precision_recall_fscore_support(df["label"].apply(lambda x:1 if x=="FAVOR" else 0),
                                               df["pred_label"].apply(lambda x:1 if x=="FAVOR" else 0), average="binary", zero_division=np.nan)
    return {"accuracy":A, "precision":P, "recall":R, "F1":F1}


if __name__ == "__main__":
    ## Get user arguments
    args = parse_args()

     ## Get labels
    favor_label,against_label = [e.strip() for e in args["labels"].split(",")]
    labels2xstance = {favor_label:"FAVOR", against_label:"AGAINST"}
    
    ## load data
    gold_data = load_xstance(args["og_data"]).set_index("id")
    pred_data = pd_read_jsonl(args["pred_data"]).set_index("id")

    ## join pred labels onto gold data
    full_data = gold_data.join(pred_data,)
    ## predicted labels to xstance default labels
    full_data["pred_label"] = full_data["pred_label"].apply(lambda x:labels2xstance[x] if x in labels2xstance else "WRONG_LABEL")
    ## evaluate globally
    res_glob = pd.DataFrame([evaluate(full_data), ])
    print(res_glob)
    ## evaluate by topic
    res_by_topic = dict()
    for t,t_df in full_data.groupby("topic"):
        res_by_topic[t] = evaluate(t_df)
    res_by_topic = pd.DataFrame(res_by_topic)
    print(res_by_topic)

    ## generate examples
    full_data["match"] = full_data["label"]==full_data["pred_label"]
    examples = full_data[["question", "comment", "topic", "label", "pred_label", "match"]]
    examples = examples[examples["match"]==False]

    ##
    if "pred_score" in full_data:
        full_data.boxplot(column="pred_score", by="match")
        plt.show()

    # ## store output file
    writer = pd.ExcelWriter(args["output_path"])
    res_glob.to_excel(writer,'Global')
    res_by_topic.to_excel(writer,'By topic')
    examples.to_excel(writer,'Examples')
    writer.close()

