import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from data import pd_read_jsonl


if __name__ == "__main__":
    og_data  = pd_read_jsonl("../data/cheese/cheese.test.jsonl")
    pred_data = pd_read_jsonl("exp1_cheese_evaluation/cheese.test.germanBert.jsonl")
    # "exp1_cheese_evaluation/cheese.test.germanBert.jsonl"
    # "exp1_cheese_evaluation/cheese.test.qwen2_5_14b.EXAMPLES_ALIGNED.jsonl"
    full_data = og_data.set_index("id").join(pred_data.set_index("id")).reset_index()
    full_data = full_data[~pd.isna(full_data["pred_label"])]
    print(full_data.columns)
    print(len(full_data))

    print(full_data[["label", "pred_label"]])

#    full_data = full_data[full_data["pred_label"].apply(lambda x: x in ["Zustimmung", "Ablehnung", "Neutral"])]

    label_2_predlabel = {"Ja, dafür":"Zustimmung",
                         "Nein, dagegen":"Ablehnung",
                         "Diskutierend":"Neutral"}
    
#    full_data["label"] = full_data["label"].apply(lambda x:label_2_predlabel[x])


#    label2id = {"Zustimmung":0, "Ablehnung":1, "Neutral":2}
    label2id = {"Ja, dafür":0, "Nein, dagegen":1, "Diskutierend":2}
    a = full_data["label"].apply(lambda x:label2id[x])
    b = full_data["pred_label"].apply(lambda x:label2id[x])

    P,R,F1,_ = precision_recall_fscore_support(a, b, average="micro")
    print("Micro")
    print("\tP={:6.2f}, R={:6.2f}, F1={:6.2f}".format(100*P, 100*R, 100*F1))

    P,R,F1,_ = precision_recall_fscore_support(a, b, average="macro")
    print("Macro")
    print("\tP={:6.2f}, R={:6.2f}, F1={:6.2f}".format(100*P, 100*R, 100*F1))
    
    P,R,F1,_ = precision_recall_fscore_support(a, b)
    print("Zustimmung")
    print("\tP={:6.2f}, R={:6.2f}, F1={:6.2f}".format(100*P[0], 100*R[0], 100*F1[0]))
    print("Ablehnung")
    print("\tP={:6.2f}, R={:6.2f}, F1={:6.2f}".format(100*P[1], 100*R[1], 100*F1[1]))
    print("Neutral")
    print("\tP={:6.2f}, R={:6.2f}, F1={:6.2f}".format(100*P[2], 100*R[2], 100*F1[2]))
