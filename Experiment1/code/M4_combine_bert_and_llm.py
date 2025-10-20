from tqdm import tqdm
import pandas as pd
import argparse
import json
import os

from data import write_jsonl, pd_read_jsonl

def parse_args():
    parser = argparse.ArgumentParser(prog='Combine predictions from LLM and GermanBert by berts prediction score.')
    parser.add_argument('llm_predictions', type=str, 
                        help='path to a jsonl file, containg the predictions from llm')
    parser.add_argument('bert_predictions', type=str, 
                        help='path to a jsonl file, containing the BERT predictions')
    parser.add_argument('output', type=str, 
                        help='path to a jsonl file, where the combined predictions will be stored')
    parser.add_argument('--threshold', type=float, default=0.9,
                        help="Prediction-Score threshold T. If prediction score is smaller than T, use LLM- else use BERT-prediciton. Defaullt=0.9")
    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()

    ## Load predicted data from BERT- and LLM-method
    llm_predictions  = pd_read_jsonl(args["llm_predictions"]).set_index("id")
    bert_predictions = pd_read_jsonl(args["bert_predictions"]).set_index("id")


    ## Convert german LLM labels to english BERT labels
    llm2bert = {"Zustimmung":"FAVOR","Ablehnung":"AGAINST"}
    llm_predictions["pred_label"] = llm_predictions["pred_label"].apply(lambda x:llm2bert[x])
    ## Combine ....
    all_predictions = bert_predictions.join(llm_predictions[["pred_label"]], lsuffix="_LLM")
    all_predictions["final_label"] = None
    #
    mask = all_predictions["pred_score"]>=args["threshold"]
    all_predictions.loc[mask, "final_label"] = all_predictions.loc[mask, "pred_label"]
    all_predictions.loc[~mask, "final_label"] = all_predictions.loc[~mask, "pred_label_LLM"]
    all_predictions["prediction_by"] = mask.apply(lambda x: "BERT" if x is True else "LLM")
    all_predictions = all_predictions[["final_label", "prediction_by"]].reset_index().rename(columns= {"final_label":"pred_label"})

    ## Store
    write_jsonl([e.to_dict() for _,e in all_predictions.iterrows()], args["output"])

