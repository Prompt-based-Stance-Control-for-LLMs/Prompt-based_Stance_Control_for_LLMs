from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
from tqdm import tqdm
import argparse
import re
import os

from data import write_jsonl, pd_read_jsonl, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(prog='Generates responses with the vanilla LLMs')
    parser.add_argument('model', type=str, 
                        help='name of the llm model to use. Only models available with ollama are valid.')
    parser.add_argument('input', type=str, 
                        help='path to a jsonl file containg input data')
    parser.add_argument('working_dir', type=str, 
                        help='path to a directory where output jsonl will be created.')
    parser.add_argument('--openai', action="store_true", default=False,
                        help='specify to use openai model')
    return vars(parser.parse_args())


if __name__ == "__main__":
    SAVE_ITER=1
    args = parse_args()

    ## Load input data
    data_df = pd_read_jsonl(args["input"])
    print("Loaded {} rows from {}".format(len(data_df), args["input"]))
    ## Create path for output file
    ds = os.path.splitext(os.path.basename((args["input"])))[0]
    ms = re.sub(r"\.|\:|\-", "_", args["model"])
    output_path = os.path.join(args["working_dir"], ds+"."+ms+".jsonl")
    print("Will store output file to: ", output_path)


    ## Build prompt template
    prompt = ChatPromptTemplate.from_messages([("human", "{user}")])
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
    total_oai_tokens = {"input_tokens":0, "output_tokens":0}
    for i,row in tqdm(data_df.iterrows(), total=len(data_df)):
        # skip if already predicted in previous checkpoint
        if row["id"] in already_finished_ids:
            continue
        # query llm
        res = chain.invoke({"user": row["prompt"]})
        if args["openai"] is True:
            total_oai_tokens["input_tokens"] += res.usage_metadata["input_tokens"]
            total_oai_tokens["output_tokens"] += res.usage_metadata["output_tokens"]
            res = res.content

        # store
        predictions.append({"id":row["id"], "topic":row["topic"], "prompt":row["prompt"], "response":res})  # note topic, must be carried to next file (for evaluation)
        # checkpoint
        if i%SAVE_ITER==0:
            write_jsonl(predictions, output_path+".checkpoint")
        # status print
        if i%10==0:
            print("[{}]  {}".format(row["topic"],row["prompt"]))
            print("-"*50)
            print(res)
            print("="*50,"\n\n")
    ## Write output to file
    write_jsonl(predictions, output_path)
    print("\nWrote predictions to {}".format(output_path))
    if args["openai"] is True:
        print("\nOpenAI-Token-Usage:", total_oai_tokens)
    
    