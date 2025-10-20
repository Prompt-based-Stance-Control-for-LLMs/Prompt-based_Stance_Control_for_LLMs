from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from tqdm import tqdm
import argparse
import os

from data import load_xstance, write_jsonl, pd_read_jsonl


def parse_args():
    parser = argparse.ArgumentParser(prog='Predict xStance data using LLM via ollama.')
    parser.add_argument('model', type=str, 
                        help='name of the llm model to use. Only models available with ollama are valid.')
    parser.add_argument('input', type=str, 
                        help='path to a jsonl file containg input data')
    parser.add_argument('output', type=str, 
                        help='path to a jsonl file, where the predictions will be saved')
    parser.add_argument('labels', type=str, 
                        help='specify FAVOR,AGAINST labels sperated by a comma. Always start with the FAVOR label! Eg. "zustimmmung,ablehnung"')
    parser.add_argument('task-prompt', type=str,
                        help='Template for defining the system prompt (task description)')
    parser.add_argument('input-template', type=str,
                        help='Template for defining the user prompt (instance input)')
    parser.add_argument('--task-as-system', action='store_true',
                        help='If specified, task-prompt will be treated as system-prompt.')
    parser.add_argument('--examples', type=str, default=None,
                        help='Optional: List of examples provided to the model (in-context learning)')
    parser.add_argument('--num-retries', type=int, default=2,
                        help="How many times should the LLM be queried if the output does not match one of the labels. Default=2")
    parser.add_argument("--cheese", action="store_true", default=False,
                        help="")
    return vars(parser.parse_args())


def load_prompt_parts(system_p, user_p, examples_p=None):
    system_t,user_t,examples = None,None,None
    with open(system_p) as ifile:
        system_t = ifile.read()
    with open(user_p) as ifile:
        user_t = ifile.read()
    if examples_p is not None:
        with open(examples_p) as ifile:
            examples = ifile.read()
    return system_t, user_t, examples


if __name__ == "__main__":
    ## Set save strategy
    SAVE_ITER = 1  # save ever n'th iteration

    ## Parse user arguments
    args = parse_args()

    ## Get labels
    allowed_labels = [e.strip() for e in args["labels"].split(",")]
    print("OUTPUT LABELS:", " | ".join(allowed_labels))

    ## Load system-, user-prompts and examples
    task_template, input_template, examples = load_prompt_parts(args["task-prompt"],
                                                                args["input-template"],
                                                                args["examples"])


    ## Build prompt template
    if args["task_as_system"] is True:
        prompt = ChatPromptTemplate.from_messages([
            ("system", task_template),
            ("human", input_template)
        ])
        print('SYSTEM PROMPT:"""\n'+task_template,'\n"""')
        print('USER PROMPT TEMPLATE:"""\n'+input_template,'\n"""')
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("human", task_template+"\n\n"+input_template)
        ])
        print('USER PROMPT TEMPLATE:"""\n'+task_template+"\n\n"+input_template,'"""\n')

    ## Define LLM pipeline
    model = OllamaLLM(model=args["model"],)
    chain = prompt | model

    ## Load data
    if args["cheese"]:
        data_df = pd_read_jsonl(args["input"])
    else:
        data_df = load_xstance(args["input"])
    print("Loaded {} instances from {}".format(len(data_df), args["input"]),"\n")

    ## check if checkpoint exists
    predictions = []
    already_finished_ids = set()
    if os.path.exists(args["output"]+".save"):
        checkpoint_data = pd_read_jsonl(args["output"]+".save")
        predictions = [e.to_dict() for _,e in checkpoint_data.iterrows()]
        already_finished_ids = set([e["id"] for e in predictions])
        print("Resuming prediction @ iteration", len(predictions))
    
    ## Predict
    print("Prediction ...")
    for i,row in tqdm(data_df.iterrows(), total=len(data_df)):
        if row["id"] in already_finished_ids:
            continue
        #
        print("="*100 + "\n" + "="*100)
        print(row["comment"])
        print("="*100)
        print("Text length: {} chars, {} words".format(len(row["comment"]), len(row["comment"].split(" "))))
        ## retry until output matches one of the labels
        retry_i = 0
        format_flag=False
        res = None
        for retry_i in range(args["num_retries"]):
            # predict current question-comment-pair
            res = chain.invoke({"question": row["question"], "text":row["comment"]})
            res = res.strip()
            # check if output is in right format
            format_flag = (res not in allowed_labels)
            #
            if (format_flag is False):
                break
            print("Wrong format @ "+str(retry_i))
        print("[{}] : (ist={}) (soll={}) (wrong_format={})".format(row["question"] , res, row["label"], format_flag))
        # store results
        predictions.append({"id":row["id"], "pred_label":res, "wrong_format":format_flag, "num_retries":retry_i})
        # checkpoint
        if i%SAVE_ITER==0:
            write_jsonl(predictions, args["output"]+".save")

    ## Write output to file
    write_jsonl(predictions, args["output"])
    print("\nWrote predictions to {}".format(args["output"]))
    
    