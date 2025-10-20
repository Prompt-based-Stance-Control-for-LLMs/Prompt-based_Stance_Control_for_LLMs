from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline    
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from tqdm import tqdm
import argparse
import json
import os

from data import load_xstance, write_jsonl, pd_read_jsonl


def parse_args():
    parser = argparse.ArgumentParser(prog='Repharse question and comment form xStance data using LLM via ollama.')
    parser.add_argument('model', type=str, 
                        help='name of the llm model to use. Only models available with ollama are valid.')
    parser.add_argument('input', type=str, 
                        help='path to a jsonl file containg input data')
    parser.add_argument('output', type=str, 
                        help='path to a jsonl file, where the predictions will be saved')
    return vars(parser.parse_args())


if __name__ == "__main__":
    ## Set save strategy
    SAVE_ITER = 1  # save ever n'th iteration

    ## Parse user arguments
    args = parse_args()

    ## Define prompt templates
    task_template = """
    Fasse die Kernaussage der folgenden Aussage in einem Satz zusammen.
    Schreibe nur die Kernaussage. Wiederhole nicht deine Aufgabe.
    Schreibe nur einen Satz.
    Schreibe nur auf deutsch.
"""
    input_template = "Aussage: {text}"
    ## Build prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", task_template),
        ("human", input_template)
    ])
    print('SYSTEM PROMPT:"""\n'+task_template,'\n"""')
    print('USER PROMPT TEMPLATE:"""\n'+input_template,'\n"""')

    ## Define LLM pipeline
    model = OllamaLLM(model=args["model"],)
    chain = prompt | model

    ## Load data
    data_df = load_xstance(args["input"])
    data_df = data_df.sample(frac=1)
    print("Loaded {} instances from {}".format(len(data_df), args["input"]),"\n")

    ## check if checkpoint exists
    predictions = []
    already_finished_ids = set()
    if os.path.exists(args["output"]+".save"):
        checkpoint_data = pd_read_jsonl(args["output"]+".save")
        predictions = [e.to_dict() for _,e in checkpoint_data.iterrows()]
        already_finished_ids = set([e["id"] for e in predictions])
        print("Resuming prediction @ iteration", len(predictions))
    
    ## Re-phrase question and text using an LLM
    print("Prediction ...")
    questions = dict()
    for i,row in tqdm(data_df.iterrows(), total=len(data_df)):
        if row["id"] in already_finished_ids:
            continue
        question = row["question"]
        text = chain.invoke({"text": row["comment"]})
        print("="*75)
        print("Question: {}\nText: {}".format(row["question"], row["comment"]))
        print("-"*50,"\n")
        print("Result: {}".format(text))

        # store results
        predictions.append({"id":row["id"], "language":"de", "comment":text, "question":row["question"], "label":row["label"]})
        # checkpoint
        if i%SAVE_ITER==0:
            write_jsonl(predictions, args["output"]+".save")

    ## Write output to file
    write_jsonl(predictions, args["output"])
    print("\nWrote predictions to {}".format(args["output"]))
    
    