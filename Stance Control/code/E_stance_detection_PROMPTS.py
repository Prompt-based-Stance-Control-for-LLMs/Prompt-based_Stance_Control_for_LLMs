from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from collections import Counter
from tqdm import tqdm
import argparse
import json
import os

from data import write_jsonl, pd_read_jsonl, load_checkpoint
from ollama_utile import classify


def parse_args():
    parser = argparse.ArgumentParser(prog='Predict xStance data using LLM via ollama.')
    parser.add_argument('model', type=str, 
                        help='name of the llm model to use. Only models available with ollama are valid.')
    parser.add_argument('input', type=str, 
                        help='path to a jsonl file containg llm responses for pro and con')
    parser.add_argument('output', type=str, 
                        help='path to a file where output jsonl will be created.')
    parser.add_argument('eval-prompt', type=str,
                        help='path to a txt file containg the system prompt for detecting the stance') 
    parser.add_argument('topics', type=str, 
                        help='path to a json file containing topic descriptions')
    parser.add_argument('--num-retries', type=int, default=5,
                        help="How many times should the LLM be queried if the output does not match one of the labels. Default=5")
    return vars(parser.parse_args())


def evaluate_full_text(text, topic, model, num_retries, allowed_classes):
    p_class,f_format,n_retry = classify(text, topic, model, num_retries, allowed_classes)
    if f_format is True:
        return None
    return p_class

def evaluate_paragraphs_text(text, topic, text_seperator, model, num_retries, allowed_classes):
    paragraphs = text.split(text_seperator)
    results = []
    for ptext in paragraphs:
        pres = evaluate_full_text(ptext, topic, model, num_retries, allowed_classes)
        results.append((ptext, pres))
    return results


if __name__ == "__main__":
    ## Set save strategy
    SAVE_ITER = 1  # save ever n'th iteration
    ## Parse user arguments
    args = parse_args()


    ## Load data
    data_df = pd_read_jsonl(args["input"])
    print("Loaded {} rows from {}".format(len(data_df), args["input"]))

    ## Load system-prompt for evaluation, and topic description
    task_path = args["eval-prompt"]
    task_template=None
    with open(task_path) as ifile:
        task_template=ifile.read()
    with open(args["topics"]) as ifile:
        topic_map = json.loads(ifile.read())
    print("Loaded stance detection prompt from:  ", task_path)
    print("Loded topic descriptions from: ", args["topics"])
    ## Create path for output file
    output_path = args["output"]
    print("Will store output file to: ", output_path)


    ## Define allowed class labels
    allowed_labels = ["Zustimmung","Ablehnung","Neutral", "Information"]
    print("Allowed output classes: "," | ".join(allowed_labels))
    ## Build prompt template
    input_template="""Frage: {question}
Text: {text}"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", task_template),
        ("human", input_template)
    ])
    ## Define LLM pipeline 
    model = OllamaLLM(model=args["model"],)
    chain = prompt | model
    input("start? press any key ...")
    ## check if checkpoint exists
    predictions, already_finished_ids = load_checkpoint(output_path+".checkpoint")
    ## Predict
    print("Prediction ...")
    for i,row in tqdm(data_df.iterrows(), total=len(data_df)):
        ## skip from previous checkpoint
        if row["id"] in already_finished_ids:
            continue
        # get topic description for current topic
        curr_topic = topic_map[row["topic"]]
        ## Create output row 
        orow = dict()
        orow["id"] = row["id"]
        orow["topic"] = row["topic"]
        orow["prompt"] = row["prompt"]
        ## llm response
        res_global = evaluate_full_text(row["prompt"], curr_topic, 
                                        chain, args["num_retries"], allowed_labels)
        orow["Prompt_Label"] = res_global
        ## store output row
        predictions.append(orow)
        ## checkpoint
        if i%SAVE_ITER==0:
            write_jsonl(predictions, output_path+".checkpoint")
        # status print
        print(row["prompt"])
        print(" =>>  ", orow["Prompt_Label"])
        print("="*50,"\n\n")

    ## Write output to file
    write_jsonl(predictions, output_path)
    print("\nWrote predictions to {}".format(output_path))
    
    