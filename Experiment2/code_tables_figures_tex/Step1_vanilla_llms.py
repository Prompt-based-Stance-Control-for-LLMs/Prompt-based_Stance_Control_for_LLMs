import pandas as pd

from data import pd_read_jsonl
from evaluation import eval_paragraph,eval_whole


STANCE_DE2EN = {"Zustimmung":"In-Favor", "Ablehnung":"Against", "Neutral":"Neutral", None:"Error", "Information":"Information"}

def evaluate_by_model(model_name, prompts_df):
    ## Load and prepare data
    gpt_vanilla_df = pd_read_jsonl("evaluation_labels/responses_vanilla.data_cleaned."+model_name+".evaluated.jsonl")
    gpt_vanilla_df = gpt_vanilla_df.set_index("id").join(prompts_df.set_index("id")).reset_index()

    ## Count stance percentages (whole response)
    global_results = eval_whole(gpt_vanilla_df["response_GlobalLabel"])
    results_by_topic = {topic:eval_whole(topic_df["response_GlobalLabel"]) for topic,topic_df in gpt_vanilla_df.groupby("topic")}
    results_by_prompt = {topic:eval_whole(topic_df["response_GlobalLabel"]) for topic,topic_df in gpt_vanilla_df.groupby("Prompt_Label")}
    
    results_by_topic = pd.DataFrame(results_by_topic)
    results_by_prompt = pd.DataFrame(results_by_prompt)

    ## Rename Columns and Index
    results_by_topic = results_by_topic.rename(columns={"EU_exit":"EU-Exit",
                                                        "immigration":"Immigration",
                                                        "social_equality":"Social-Equality"})
    results_by_topic.index = [STANCE_DE2EN[i] for i in results_by_topic.index]
    results_by_topic = results_by_topic.transpose().reset_index().rename(columns={"index":"Topic"})

    ## Rename Columns and Index
    results_by_prompt = results_by_prompt.rename(columns=STANCE_DE2EN)
    results_by_prompt.index = [STANCE_DE2EN[i] for i in results_by_prompt.index]
    results_by_prompt = results_by_prompt.transpose().reset_index().rename(columns={"index":"Prompt-Label"})
    
    ##
    return global_results, results_by_topic, results_by_prompt


def output_latex(results, by_topic, by_prompt):
    o = ""
    o += "% Vanilla Stances (Whole response)\n\n"
    o += results.set_index("Model")[["In-Favor", "Against", "Neutral", "Error"]].to_latex(float_format="{:.2f}".format)+"\n\n\n"
    o += "% Vanilla Stances (Whole response) by topic\n\n"
    o += by_topic.set_index(["Model", "Topic"])[["In-Favor", "Against", "Neutral", "Error"]].to_latex(float_format="{:.2f}".format)+"\n\n\n"
    o += "% Vanilla Stances (Whole response) by prompt\n\n"
    o += by_prompt.set_index(["Model", "Prompt-Label"])[["In-Favor", "Against", "Neutral", "Error"]].to_latex(float_format="{:.2f}".format)+"\n\n\n"
    return o


if __name__ == "__main__":
    ## Load user message data (with stance-labels)
    prompts_df = pd_read_jsonl("data/data_cleaned.PromptStances.SpecialSystemPrompt.jsonl")
    prompts_df = prompts_df[["id", "Prompt_Label"]]

    ## Count percentages
    gpt_res, gpt_by_topic, gpt_by_prompt = evaluate_by_model("gpt_3_5_turbo", prompts_df)
    mistral_res, mistral_by_topic, mistral_by_prompt = evaluate_by_model("mistral_small", prompts_df)
    gemma_res, gemma_by_topic, gemma_by_prompt = evaluate_by_model("gemma3_4b", prompts_df)

    ## Merge models into one table
    gpt_by_topic["Model"],gpt_by_prompt["Model"] = "GPT-3.5-turbo","GPT-3.5-turbo"
    mistral_by_topic["Model"],mistral_by_prompt["Model"] = "Mistral-Smalll","Mistral-Small"
    gemma_by_topic["Model"],gemma_by_prompt["Model"] = "Gemma3-4b","Gemma3-4b"
    #
    results = pd.DataFrame({"GPT-3.5-turbo":gpt_res, "Mistral-Small":mistral_res, "Gemma3-4b":gemma_res}).fillna(0).transpose()
    results = results.reset_index().rename(columns={"index":"Model"}).rename(columns=STANCE_DE2EN)
    results_by_topic = pd.concat((gpt_by_topic, mistral_by_topic, gemma_by_topic)).fillna(0)
    results_by_prompt = pd.concat((gpt_by_prompt, mistral_by_prompt, gemma_by_prompt)).fillna(0)

    ##
    print("Vanilla reponse stances:")
    print(results)
    print("\nBy Topic:")
    print(results_by_topic.set_index(["Model", "Topic"]))
    print("\nBy Prompt:")
    print(results_by_prompt.set_index(["Model", "Prompt-Label"]))

    ##
    o = output_latex(results, results_by_topic, results_by_prompt).replace("_", r"\_")
    with open("results/step1_vanilla_llms.tex", "w") as ofile:
        ofile.write(o)