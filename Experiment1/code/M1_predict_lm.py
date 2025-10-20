from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import argparse

from data import load_xstance, xstance_instance2text, write_jsonl, pd_read_jsonl

def parse_args():
    parser = argparse.ArgumentParser(prog='Finetune a LLM from huggingface on XStance data.')
    parser.add_argument('model_type', type=str, 
                        help='huggingface model path')
    parser.add_argument('model_path', type=str, 
                        help='local path to fine-tuned model')
    parser.add_argument('data_input', type=str, 
                        help='path to a jsonl file containg test data to be predicted')
    parser.add_argument('data_output', type=str, 
                        help='filepath where predictions will be stored')
    parser.add_argument("--cheese", action="store_true", default=False,
                        help="")
    return vars(parser.parse_args())


if __name__ == "__main__":
    ## Parse user input arguments
    args = parse_args()

    ## Load fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(args["model_type"], padding="max_length", truncation=True,)  # model_key
    fine_tuned_model = AutoModelForSequenceClassification.from_pretrained(args["model_path"], model_type=args["model_type"])
    fine_tuned_model.eval()  # frezze model params
    clf = pipeline("text-classification", model=fine_tuned_model, tokenizer=tokenizer)
    ## Load data
    if args["cheese"]:
        test_df = pd_read_jsonl(args["data_input"])
    else:
        test_df = load_xstance(args["data_input"])
    print("Loaded {} instances from {}".format(len(test_df), args["data_input"]),"\n")
    test_df["model_input"] = xstance_instance2text(test_df)

    ## Predict test data
    print("Predicting data...")
    results = clf(test_df["model_input"].to_list())
    test_df["pred_label"] = [e["label"] for e in results]
    test_df["pred_score"] = [e["score"] for e in results]

    ## Store predictions
    output_data = [e.to_dict() for _,e in test_df[["id", "pred_label", "pred_score"]].iterrows()]
    write_jsonl(output_data, args["data_output"])
    print("\nWrote predictions to {}".format(args["data_output"]))
