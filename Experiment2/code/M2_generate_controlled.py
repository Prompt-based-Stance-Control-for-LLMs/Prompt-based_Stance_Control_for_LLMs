from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
from tqdm import tqdm
import argparse
import json
import os
import re

from data import write_jsonl, pd_read_jsonl, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(prog='Generate responses with controlled LLMs.')
    parser.add_argument('model', type=str, 
                        help='name of the llm model to use. Only models available with ollama are valid.')
    parser.add_argument('input', type=str, 
                        help='path to a jsonl file containg input data')
    parser.add_argument('working_dir', type=str, 
                        help='path to a directory where output jsonl will be created. System prompts are expected to be stored there.')
    parser.add_argument('topics',  type=str, 
                        help='path to a json file, containing the topic descriptions') 
    parser.add_argument('--openai', action="store_true", default=False,
                        help='specify to use openai model')
    return vars(parser.parse_args())


if __name__ == "__main__":
    SAVE_ITER=1
    args = parse_args()

            
    ## Load input data
    data_df = pd_read_jsonl(args["input"])
    print("Loaded {} rows from {}".format(len(data_df), args["input"]))

    ## Load system prompts from working dir
    favor_path = os.path.join(args["working_dir"], "system_infavor.txt")
    against_path = os.path.join(args["working_dir"], "system_against.txt")
    #
    with open(favor_path) as ifile:
        system_pro = ifile.read()
    with open(against_path) as ifile:
        system_con = ifile.read()
    with open(args["topics"]) as ifile:
        topic_map = json.loads(ifile.read())
    print("Loded system prompt (infavor) from: ", favor_path)
    print("Loded system prompt (against) from: ", against_path)
    print("Loded topic descriptions from: ", args["topics"])
    ## Create path for output file
    ds = os.path.splitext(os.path.basename((args["input"])))[0]
    ms = re.sub(r"\.|\:|\-", "_", args["model"])
    output_path = os.path.join(args["working_dir"], ds+"."+ms+".jsonl")
    print("Will store output file to: ", output_path)

    
    ## Build prompt template, from input files
    msgs = []
    prompt_pro = ChatPromptTemplate.from_messages([
        ("system", system_pro),
        ("human", "{user}")
    ])
    prompt_con = ChatPromptTemplate.from_messages([
        ("system", system_con),
        ("human", "{user}")
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
    chain_pro = prompt_pro | model
    chain_con = prompt_con | model

    ## check if checkpoint exists
    predictions,already_finished_ids = load_checkpoint(output_path+".checkpoint")
    ## Predict
    total_oai_tokens = {"input_tokens":0, "output_tokens":0}
    for i,row in tqdm(data_df.iterrows(), total=len(data_df)):
        # skip if already predicted in previous checkpoint
        if row["id"] in already_finished_ids:
            continue
        # map topic to proper language
        topic = topic_map[row["topic"]]
        # query LLM with both types of prompts (pro and con)
        ans_pro = chain_pro.invoke({"user":row["prompt"], "topic":topic})
        if args["openai"] is True:
            total_oai_tokens["input_tokens"] += ans_pro.usage_metadata["input_tokens"]
            total_oai_tokens["output_tokens"] += ans_pro.usage_metadata["output_tokens"]
            ans_pro = ans_pro.content
        ans_con = chain_con.invoke({"user":row["prompt"], "topic":topic})
        if args["openai"] is True:
            total_oai_tokens["input_tokens"] += ans_con.usage_metadata["input_tokens"]
            total_oai_tokens["output_tokens"] += ans_con.usage_metadata["output_tokens"]
            ans_con = ans_con.content
        # store results
        predictions.append({"id":row["id"], "topic":row["topic"], "prompt":row["prompt"], "infavor_response":ans_pro, "against_response":ans_con})
        # checkpoint
        if i%SAVE_ITER==0:
            write_jsonl(predictions, output_path+".checkpoint")
        # status print
        if i%10==0:
            print("[{}]  {}".format(topic,row["prompt"]))
            print("-"*50)
            print(ans_pro)
            print("-"*50)
            print(ans_con)
            print("="*50,"\n\n")
    ## Write output to file
    write_jsonl(predictions, output_path)
    print("\nWrote predictions to {}".format(output_path))
    if args["openai"] is True:
        print("\nOpenAI-Token-Usage:", total_oai_tokens)
    
    