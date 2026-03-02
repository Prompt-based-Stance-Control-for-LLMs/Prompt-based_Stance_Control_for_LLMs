import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15) 
matplotlib.rc('legend', fontsize=16 )
matplotlib.rc('axes', titlesize=18, labelsize=18)
matplotlib.rc('figure', figsize=(12,10), dpi=180)
import pandas as pd
from tqdm import tqdm
import re

from data import pd_read_jsonl


if __name__ == "__main__":
    model_name = "gemma3_4b"
    vanilla_df  = pd_read_jsonl("evaluation_labels/responses_vanilla.data_cleaned."+model_name+".evaluated.jsonl")
    basic_df    = pd_read_jsonl("evaluation_labels/responses_basic.data_cleaned."+model_name+".evaluated.jsonl")
    extended_df = pd_read_jsonl("evaluation_labels/responses_noNumbering.data_cleaned."+model_name+".evaluated.jsonl")

    vanilla_errors = vanilla_df[pd.isna(vanilla_df["response_GlobalLabel"])]
    basic_i_errors = basic_df[pd.isna(basic_df["infavor_response_GlobalLabel"])]
    basic_a_errors = basic_df[pd.isna(basic_df["against_response_GlobalLabel"])]
    extended_i_errors = extended_df[pd.isna(extended_df["infavor_response_GlobalLabel"])]
    extended_a_errors = extended_df[pd.isna(extended_df["against_response_GlobalLabel"])]

    print("Vanilla:", len(vanilla_errors))
    print("Basic-I:", len(basic_i_errors))
    print("Basic-A:", len(basic_a_errors))
    print("Extended-I:", len(extended_i_errors))
    print("Extended-A:", len(extended_a_errors))
    input()

    for _,row in vanilla_errors.iterrows():
        print("="*100)
        print("="*100)
        for e in row["response_ParagraphsLabeled"]:
            print(e[0][:min(len(e[0]), 1000)])
            print("*"*100)
            print("*"*100)
        input()
    for _,row in list(basic_i_errors.iterrows())+list(extended_i_errors.iterrows()):
        for e in row["infavor_response_ParagraphsLabeled"]:
            print(e[0][:min(len(e[0]), 1000)])
            print("*"*100)
            print("*"*100)
        input()
    for _,row in list(basic_a_errors.iterrows())+list(extended_a_errors.iterrows()):
        for e in row["against_response_ParagraphsLabeled"]:
            print(e[0][:min(len(e[0]), 1000)])
            print("*"*100)
            print("*"*100)
        input()