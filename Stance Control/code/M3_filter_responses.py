from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
from tqdm import tqdm
import argparse
import json
import os

from data import write_jsonl, pd_read_jsonl, load_checkpoint
from ollama_utile import classify


def parse_args():
    parser = argparse.ArgumentParser(prog='Filteres paragraphs not matching the target stance from given LLM responses.')
    parser.add_argument('model', type=str, 
                        help='name of the llm model to use for detecting stance. Only models available with ollama are valid.')
    parser.add_argument('input', type=str, 
                        help='path to a jsonl file containg llm responses for pro and con')
    parser.add_argument('working_dir', type=str, 
                        help='path to a directory where output jsonl will be created.')
    parser.add_argument('topics',  type=str, 
                        help='path to a json file, containing the topic descriptions') 
    parser.add_argument('eval-prompt', type=str,
                        help='path to a txt file containg the system prompt for detecting the stance') 
    parser.add_argument('--num-retries', type=int, default=5,
                        help="How many times should the LLM be queried if the output does not match one of the labels. Default=5")
    parser.add_argument('--openai', action="store_true", default=False,
                        help='specify to use openai model')
    return vars(parser.parse_args())


def tcolor(t, c):
    if c=="green":
        return u"\u001b[32m"+t+u"\u001b[0m"
    if c=="yellow":
        return u"\u001b[33m"+t+u"\u001b[0m"
    if c=="red":
        return u"\u001b[31m"+t+u"\u001b[0m"
    if c=="blue":
        return u"\u001b[36m"+t+u"\u001b[0m"


def filter_list_of_texts(texts, topic, class_to_filter, model, num_retries, allowed_classes, total_oai_tokens=None):
    otext = list(texts)
    removed_text = []
    # print("="*50)
    # print("Removing: ", class_to_filter)
    # print("="*25)
    for i in range(len(otext)-1,-1,-1):
        ptext = otext[i]
        ## Classify using LLM
        result, wrong_format, nretries = classify(ptext, topic, 
                                                  model, num_retries, allowed_classes, total_oai_tokens)
        # print(ptext, "   ===>   ", tcolor(result, "green" if result=="Zustimmung" else ("red" if result=="Ablehnung" else ("yellow" if result=="Neutral" else "blue")) ))
        # print()
        ## Skip if label was not correct
        if wrong_format:
            continue
        ## Delte paragraph if not in line with target stance
        if result==class_to_filter:
            removed_text.append(otext[i])
            del otext[i]
    # print("*"*50)
    ##
    return otext,removed_text



if __name__ == "__main__":
    ## Set save strategy
    SAVE_ITER = 1  # save ever n'th iteration
    ## Parse user arguments
    args = parse_args()

    ## Load data
    data_df = pd_read_jsonl(args["input"])
    print("Loaded {} rows from {}".format(len(data_df), args["input"]))
    ## Load system-prompt for evaluation and topic descriptions
    task_template=None
    with open(args["eval-prompt"]) as ifile:
        task_template=ifile.read()
    with open(args["topics"]) as ifile:
        topic_map = json.loads(ifile.read())
    print("Loaded stance detection prompt from:  ", args["eval-prompt"])
    print("Loded topic descriptions from: ", args["topics"])
    ## Create path for output file
    fn_in = os.path.splitext(os.path.basename((args["input"])))[0]
    output_path = os.path.join(args["working_dir"], fn_in+".filtered.jsonl")
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
    if args["openai"] is True:
        model = ChatOpenAI(
            model=args["model"],
            max_tokens=None,
            timeout=None,
            max_retries=2,)
    else:
        model = OllamaLLM(model=args["model"])
    chain = prompt | model


    ## check if checkpoint exists
    predictions,already_finished_ids = load_checkpoint(output_path+".checkpoint")
    ## Predict
    total_oai_tokens = None
    if args["openai"] is True:
        total_oai_tokens = {"input_tokens":0, "output_tokens":0}
    for j,row in tqdm(data_df.iterrows(), total=len(data_df)):
        # skip if already predicted in previous checkpoint
        if row["id"] in already_finished_ids:
            continue
        # map topic to proper language
        curr_topic = topic_map[row["topic"]]
        ## FILTER PRO RESULTS

        pro_paragraphs = row["infavor_response"].split("\n\n")
        pro_paras_filtered,pro_removed = filter_list_of_texts(pro_paragraphs, curr_topic, "Ablehnung", 
                                                              chain, args["num_retries"], allowed_labels, total_oai_tokens)
        ## FILTER CON RESULTS
        con_paragraphs = row["against_response"].split("\n\n")
        con_paras_filtered,con_removed = filter_list_of_texts(con_paragraphs, curr_topic, "Zustimmung", 
                                                              chain, args["num_retries"], allowed_labels, total_oai_tokens)
        
        ## Store results
        predictions.append({"id":row["id"],
                            "topic":row["topic"],
                            "prompt":row["prompt"],
                            "infavor_response":"\n\n".join(pro_paras_filtered),
                            "against_response":"\n\n".join(con_paras_filtered),
                            "infavor_removed_paragraphs":pro_removed,
                            "against_removed_paragraphs":con_removed
                            })
        # checkpoint
        if j%SAVE_ITER==0:
            write_jsonl(predictions, output_path+".checkpoint")
        # status print
        if j%10==0:
            print("================================================= Infavor-Response:")
            print(row["infavor_response"])
            print("*"*25, "Removed from Infavor-Response", "-"*25)
            print("\n".join(pro_removed))
            print("\n\n================================================= Against-Response:")
            print(row["against_response"])
            print("*"*25, "Removed from Against-Response", "-"*25)
            print("\n".join(con_removed))
            print("="*79,"\n\n")

    ## Write output to file
    write_jsonl(predictions, output_path)
    print("\nWrote predictions to {}".format(output_path))
    if args["openai"] is True:
        print("\nOpenAI-Token-Usage:", total_oai_tokens)