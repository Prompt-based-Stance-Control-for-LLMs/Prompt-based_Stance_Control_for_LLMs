from tqdm import tqdm
import argparse

from data import pd_read_jsonl, write_jsonl, load_xstance

def parse_args():
    parser = argparse.ArgumentParser(prog='Convert predictions from FastText baseline into same format as my own predictions')
    parser.add_argument('original_file', type=str, 
                        help='path to original data (eg. valid.jsonl)')
    parser.add_argument('pred_file', type=str, 
                        help='path to the predicted file (eg. valid.pred.jsonl)')
    parser.add_argument('output_path', type=str, 
                        help='jsonl path where converted predictions should be stored')
    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()
    
    og_file = load_xstance(args["original_file"])
    p_file = pd_read_jsonl(args["pred_file"])
    assert(len(og_file)==len(p_file))
    full_data = []
    for (_,orow),(_,prow) in zip(og_file.iterrows(), p_file.iterrows()):
        full_data.append({"id":orow["id"], "pred_label":prow['label']})

    write_jsonl(full_data, args["output_path"])