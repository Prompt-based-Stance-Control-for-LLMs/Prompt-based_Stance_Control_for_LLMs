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
    parser.add_argument('working_dir', type=str, 
                        help='path to a directory where output jsonl will be created.')
    parser.add_argument('topics', type=str, 
                        help='path to a json file containing topic descriptions')
    parser.add_argument('--filtered', type=str, default=None,
                        help="Path to a previous evaluation, will re-evaluate only the rows from this file that have changes")
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

def take_previous_paragraphs(row, key, text_seperator, paragraph_file):
    previous_paragraphs = paragraph_file.loc[paragraph_file["id"]==row["id"], key+"_ParagraphsLabeled"].item()
    previous_paragraphs = {k:v for k,v in previous_paragraphs}
    paragraphs = row[key].split(text_seperator)
    results = []
    for ptext in paragraphs:
        if ptext in previous_paragraphs:
            results.append((ptext, previous_paragraphs[ptext]))
    print(len(previous_paragraphs), len(results))
    return results

if __name__ == "__main__":
    ## Set save strategy
    SAVE_ITER = 1  # save ever n'th iteration
    ## Parse user arguments
    args = parse_args()


    ## Load data
    data_df = pd_read_jsonl(args["input"])
    print("Loaded {} rows from {}".format(len(data_df), args["input"]))
    if args["filtered"]:
        filtered_og_file = pd_read_jsonl(args["filtered"])
        print("Loded {} rows from paragraph file {}".format(len(filtered_og_file), args["filtered"]))

    ## Load system-prompt for evaluation, and topic description
    task_path = os.path.join(args["working_dir"],"system.txt")
    task_template=None
    with open(task_path) as ifile:
        task_template=ifile.read()
    with open(args["topics"]) as ifile:
        topic_map = json.loads(ifile.read())
    print("Loaded stance detection prompt from:  ", task_path)
    print("Loded topic descriptions from: ", args["topics"])

    ## Create path for output file
    filename_in = os.path.splitext(os.path.basename((args["input"])))[0]
    folder_in = os.path.split(os.path.split(args["input"])[0])[1]
    fn_out = folder_in+"."+filename_in+".evaluated.jsonl"
    output_path = os.path.join(args["working_dir"], fn_out)
    print("Will store output file to: ", output_path)


    ## Define allowed class labels
    allowed_labels = ["Zustimmung","Ablehnung","Neutral"]
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

        ## If an original file was provided
        if args["filtered"] is not None:
            ## Check if row was filtered at all
            np_filtered_i = len(row["infavor_removed_paragraphs"])
            np_filtered_a = len(row["against_removed_paragraphs"])
            ## Get og row from which was filtered
            og_row = filtered_og_file[filtered_og_file["id"]==row["id"]]
            ## If infavor response was not filtered
            if np_filtered_i==0:
                print("Copying infavor ...")
                orow["infavor_response_GlobalLabel"]        = og_row["infavor_response_GlobalLabel"].item()
                orow["infavor_response_ParagraphsLabeled"]  = og_row["infavor_response_ParagraphsLabeled"].item()
            else:
                print("Evaluating infavor ...")
                orow["infavor_response_GlobalLabel"] = evaluate_full_text(row["infavor_response"], curr_topic, 
                                                                          chain, args["num_retries"], allowed_labels)
                orow["infavor_response_ParagraphsLabeled"] = take_previous_paragraphs(row, "infavor_response", "\n\n", filtered_og_file)
            ## If against response was not filtered
            if np_filtered_a==0:
                print("Copying against ...")
                orow["against_response_GlobalLabel"]        = og_row["against_response_GlobalLabel"].item()
                orow["against_response_ParagraphsLabeled"]  = og_row["against_response_ParagraphsLabeled"].item()
            else:
                print("Evaluating against ...")
                orow["against_response_GlobalLabel"] = evaluate_full_text(row["against_response"], curr_topic, 
                                                                          chain, args["num_retries"], allowed_labels)
                orow["against_response_ParagraphsLabeled"] = take_previous_paragraphs(row, "against_response", "\n\n", filtered_og_file)
        ## If not original file was provided
        else:
            ## llm response
            for key in row.keys():
                if "response" in key:
                    res_global = evaluate_full_text(row[key], curr_topic, 
                                                    chain, args["num_retries"], allowed_labels)
                    res_paragraphs = evaluate_paragraphs_text(row[key], curr_topic, "\n\n",
                                                              chain, args["num_retries"], allowed_labels)
                    orow[key+"_GlobalLabel"] = res_global
                    orow[key+"_ParagraphsLabeled"] = res_paragraphs
        ## store output row
        predictions.append(orow)
        print("======================================")
        ## checkpoint
        if i%SAVE_ITER==0:
            write_jsonl(predictions, output_path+".checkpoint")
        # status print
        if i%10==0:
            for key in orow.keys():
                if "_GlobalLabel" in key:
                    print("{}: {}".format(key, orow[key]))
            print("="*50,"\n\n")

    ## Write output to file
    write_jsonl(predictions, output_path)
    print("\nWrote predictions to {}".format(output_path))
    
    