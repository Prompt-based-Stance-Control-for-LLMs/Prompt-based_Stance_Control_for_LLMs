import pandas as pd
import json
import os 


def pd_read_jsonl(path):
    data = []
    with open(path, "r") as ifile:
        for line in ifile:
            if line.strip()=="":
                continue
            data.append(json.loads(line))
    return pd.DataFrame(data)


def write_jsonl(list_of_dicts, opath):
    with open(opath, "w") as ofile:
        for d in list_of_dicts:
            ofile.write(json.dumps(d)+"\n")


def prepare_data(xlsx_path, opath):
    data_df = pd.read_excel(xlsx_path)
    rows = []
    counter=0
    for _,row in data_df.iterrows():
        rows.append({
            "id":counter,
            "author_id":row["id"],
            "topic":"immigration",
            "prompt":row["discussion_01"]
            })
        rows.append({
            "id":counter+1,
            "author_id":row["id"],
            "topic":"EU_exit",
            "prompt":row["discussion_02"]
        })
        rows.append({
            "id":counter+2,
            "author_id":row["id"],
            "topic":"social_equality",
            "prompt":row["discussion_03"]
        })
        counter += 3
    write_jsonl(rows, opath) 


def split_valid_test(jsonl_path, odir, val_ratio):
    data = pd_read_jsonl(jsonl_path)

    valid = data.groupby("topic").sample(int(len(data)*val_ratio/3))
    test  = data[data["id"].apply(lambda x:x not in valid["id"].to_list())]

    print(valid)
    print(test)
    print(len(test), "+", len(valid), "=", len(data))

    write_jsonl([e.to_dict() for _,e in valid.iterrows()], os.path.join(odir, "valid.jsonl"))
    write_jsonl([e.to_dict() for _,e in test.iterrows()], os.path.join(odir, "test.jsonl"))


def load_checkpoint(file_path):
    already_finished_ids = set()
    predictions = []
    if os.path.exists(file_path):
        checkpoint_data = pd_read_jsonl(file_path)
        predictions = [e.to_dict() for _,e in checkpoint_data.iterrows()]
        already_finished_ids = set([e["id"] for e in predictions])
        print("Resuming prediction @ iteration", len(predictions))
    return predictions, already_finished_ids