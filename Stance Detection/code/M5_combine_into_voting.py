from collections import Counter

from data import pd_read_jsonl, write_jsonl


def argmax(ddict):
    max_val,max_key = 0,None
    for k,v in ddict.items():
        if v>max_val:
            max_val=v
            max_key=k
    return max_key

if __name__=="__main__":
    english_predictions = [
        "experiment_1/exp1_german_bert/6348.test.predictions.jsonl",
    ]
    german_predictions = [
        "experiment_1/exp1_german_example/test.qwen_72b.jsonl",
        "experiment_1/exp1_german_simple/test.sauerkraut.jsonl",
    ]

    output_path = "experiment_1/exp1_voted/test.voted.predictions.jsonl"
    ## load all english prediction files
    pred_data = []
    for fpath in english_predictions:
        data = pd_read_jsonl(fpath)
        data["pred_label"] = data["pred_label"].apply(lambda x:x)
        data = data[["id", "pred_label"]].rename(columns={"pred_label":fpath})
        pred_data.append( data )
    ## load all german prediction files
    german2english = {"Zustimmung":"FAVOR", "Ablehnung":"AGAINST"}
    for fpath in german_predictions:
        data = pd_read_jsonl(fpath)
        data["pred_label"] = data["pred_label"].apply(lambda x:german2english[x] if x in german2english else x)
        data = data[["id", "pred_label"]].rename(columns={"pred_label":fpath})
        pred_data.append( data )
    
    ## merge all prediction files into one dataframe
    full_data = pred_data[0]
    full_data = full_data.set_index("id")
    for data in pred_data[1:]:
        full_data = full_data.join(data.set_index("id"))
    print(full_data)
    ## select most voted label
    final_pred_data = []
    for id,row in full_data.iterrows():
        label_counts = Counter(row.values)
        voted_label = argmax(label_counts)
        print(label_counts,"==>",voted_label)
        final_pred_data.append({"id":id, "pred_label":voted_label})
    ## write output file
    write_jsonl(final_pred_data, output_path)
    