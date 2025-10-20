from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import argparse
import datasets
import os

from data import load_xstance, xstance_instance2text, pd_read_jsonl


def parse_args():
    parser = argparse.ArgumentParser(prog='Finetune a LLM from huggingface on XStance data.')
    parser.add_argument('model', type=str, 
                        help='huggingface model path ')
    parser.add_argument('train', type=str, 
                        help='path to a jsonl file containg training data')
    parser.add_argument('dev', type=str, 
                        help='path to a jsonl file containg development data')
    parser.add_argument('output', type=str, 
                        help='folder path where to store the model checkpoints and final model')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs to train. Default=3')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch-size during training. Default=16')
    parser.add_argument('--warmup-ratio', type=float, default=0.1,
                        help='Fraction of training iterations used for warm-up (increasing lr). Default=0.1')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Weight-decay parameter for AdamW optimizer (regularization strength). Default=0.0')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate to start training with. Default=0.00002')
    parser.add_argument("--cheese", action="store_true", default=False,
                        help="")
    return vars(parser.parse_args())




## load and prepare data
def load_n_prepare_data(train_path, dev_path, tokenizer, label2id, MAX_SEQ_LENGTH=512, cheese=False):
    ## Load data
    print("Load and prepare data ...")
    if cheese:
        train_df = pd_read_jsonl(train_path)
        valid_df = pd_read_jsonl(dev_path)
    else:
        train_df = load_xstance(train_path)
        valid_df = load_xstance(dev_path)
    
    print(train_df["comment"].apply(lambda x:len(x.split(" "))).max())
    print(valid_df["comment"].apply(lambda x:len(x.split(" "))).max())


    # convert to a huggingface dataset
    train_ds = datasets.Dataset.from_pandas(train_df)
    valid_ds = datasets.Dataset.from_pandas(valid_df)
    # prepare data (append text to target, convert labels)
    def prepare_instances(example):
        input_text = xstance_instance2text(example)
        input_label = label2id[example["label"]]
        return {"text":input_text, "label":input_label}
    train_ds = train_ds.map(prepare_instances)
    valid_ds = valid_ds.map(prepare_instances)
    # tokenize data
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH)
    tokenized_train = train_ds.map(tokenize_function, batched=True).shuffle(seed=42)
    tokenized_valid = valid_ds.map(tokenize_function, batched=True).shuffle(seed=42)
    #
    return tokenized_train, tokenized_valid

## Define evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall}


if __name__ == "__main__":
    ## Parse user input arguments
    args = parse_args()

    ## Define labels
    if args["cheese"] is True:
        labels = ["Diskutierend", "Ja, dafür", "Nein, dagegen", "Unklar"] # ["Zustimmung", "Neutral", "Ablehnung"]
    else:
        labels = ["FAVOR", "AGAINST"]
    label2id = {l:i for i,l in enumerate(labels)}
    id2label = {v:k for k,v in label2id.items()}

    ## Define model
    tokenizer = AutoTokenizer.from_pretrained(args["model"])
    model = AutoModelForSequenceClassification.from_pretrained(args["model"], 
                                                               problem_type="single_label_classification", 
                                                               num_labels=len(labels),
                                                               id2label=id2label,
                                                               label2id=label2id)
    ## Load and prepare data
    print("Preparing data ...")
    tokenized_train, tokenized_valid = load_n_prepare_data(args["train"], args["dev"], tokenizer, label2id, MAX_SEQ_LENGTH=512, cheese=args["cheese"])
    ## Define training
    training_args = TrainingArguments(output_dir=args["output"], 
                                      eval_strategy="epoch",
                                      save_strategy="epoch",
                                      warmup_ratio=args["warmup_ratio"],
                                      num_train_epochs=args["epochs"],
                                      learning_rate=args["lr"],
                                      weight_decay=args["weight_decay"],
                                      per_device_train_batch_size=args["batch_size"])
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        compute_metrics=compute_metrics,
    )

    ## Start Train
    print("Fitting model ...")
    trainer.train()
    trainer.save_model(os.path.join(args["output"], "final_model"))  
