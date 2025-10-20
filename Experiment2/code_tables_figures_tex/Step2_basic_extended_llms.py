import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15) 
matplotlib.rc('legend', fontsize=16 )
matplotlib.rc('axes', titlesize=18, labelsize=18)
matplotlib.rc('figure', figsize=(12,10), dpi=180)
import pandas as pd

from data import pd_read_jsonl
from evaluation import eval_paragraph,eval_whole


STANCE_DE2EN = {"Zustimmung":"In-Favor", "Ablehnung":"Against", "Neutral":"Neutral", None:"Error", "Information":"Information"}

def evaluate_by_model(model_name):
    ## Load and prepare data
    basic_df = pd_read_jsonl("evaluation_labels/responses_basic.data_cleaned."+model_name+".evaluated.jsonl")
    extended_df = pd_read_jsonl("evaluation_labels/responses_noNumbering.data_cleaned."+model_name+".evaluated.jsonl")
    
    ## Count stance percentages (whole response)
    basic_infavor = eval_whole(basic_df["infavor_response_GlobalLabel"])
    basic_against = eval_whole(basic_df["against_response_GlobalLabel"])
    extended_infavor = eval_whole(extended_df["infavor_response_GlobalLabel"])
    extended_against = eval_whole(extended_df["against_response_GlobalLabel"])
    
    new_table = [
        dict([("Prompt","Basic"), ("Tgt.Stance","infavor"),]+list(basic_infavor.items())),
        dict([("Prompt","Basic"), ("Tgt.Stance","against"),]+list(basic_against.items())),
        dict([("Prompt","Extended"), ("Tgt.Stance","infavor"),]+list(extended_infavor.items())),
        dict([("Prompt","Extended"), ("Tgt.Stance","against"),]+list(extended_against.items())),
    ]
    results = pd.DataFrame(new_table)

    ##
    results = results.rename(columns=STANCE_DE2EN)
    results = results.fillna(0)
    ##
    return results

def evaluate_by_model_and(model_name, by):
    ## Load user message data (with stance-labels)
    prompts_df = pd_read_jsonl("data/data_cleaned.PromptStances.SpecialSystemPrompt.jsonl")
    prompts_df = prompts_df[["id", "Prompt_Label"]]
    ## Load and prepare data
    basic_df = pd_read_jsonl("evaluation_labels/responses_basic.data_cleaned."+model_name+".evaluated.jsonl")
    extended_df = pd_read_jsonl("evaluation_labels/responses_noNumbering.data_cleaned."+model_name+".evaluated.jsonl")
    #
    basic_df = basic_df.set_index("id").join(prompts_df.set_index("id")).reset_index()
    extended_df = extended_df.set_index("id").join(prompts_df.set_index("id")).reset_index()
    
    ## Count stance percentages (whole response)
    new_table = list()
    for gid, gdf in basic_df.groupby(by):
        basic_infavor = eval_whole(gdf["infavor_response_GlobalLabel"])
        basic_against = eval_whole(gdf["against_response_GlobalLabel"])
        new_table.extend([
                dict([("Prompt","Basic"), (by, gid), ("Tgt.Stance","infavor"),]+list(basic_infavor.items())),
                dict([("Prompt","Basic"), (by, gid), ("Tgt.Stance","against"),]+list(basic_against.items())),
        ])
    for gid, gdf in extended_df.groupby(by):
        extended_infavor = eval_whole(gdf["infavor_response_GlobalLabel"])
        extended_against = eval_whole(gdf["against_response_GlobalLabel"])
        new_table.extend([
            dict([("Prompt","Extended"), (by, gid), ("Tgt.Stance","infavor"),]+list(extended_infavor.items())),
            dict([("Prompt","Extended"), (by, gid), ("Tgt.Stance","against"),]+list(extended_against.items())),
        ])
    results = pd.DataFrame(new_table)
    ##
    results = results.rename(columns=STANCE_DE2EN)
    results = results.fillna(0)
    ##
    return results

def post_process_tables(results_df, topic_df, prompt_df, model_name):
    ##
    topic_df = topic_df.rename(columns={"topic":"Topic"})
    topic_df["Topic"] = topic_df["Topic"].apply(lambda x:{"immigration":"Immigration",
                                                          "EU_exit":"EU-Exit",
                                                          "social_equality":"Social-Equality"}[x])
    ##
    prompt_df = prompt_df.rename(columns={"Prompt_Label":"Prompt-Lbl."})
    prompt_df["Prompt-Lbl."] = prompt_df["Prompt-Lbl."].apply(lambda x:STANCE_DE2EN[x])
    ##
    results_df["Model"] = model_name
    topic_df["Model"] = model_name
    prompt_df["Model"] = model_name
    ##
    return results_df,topic_df, prompt_df

def make_match_table(data_df):
    ##
    mask_i = data_df["Tgt.Stance"] == "infavor"
    mask_a = data_df["Tgt.Stance"] == "against"
    #
    data_df["Match"] = None
    data_df["No-Match"] = None
    data_df.loc[mask_i, "Match"] = data_df.loc[mask_i, "In-Favor"]
    data_df.loc[mask_i, "No-Match"] = (data_df.loc[mask_i, "Against"] + data_df.loc[mask_i, "Neutral"] + data_df.loc[mask_i, "Error"])
    #
    data_df.loc[mask_a, "Match"] = data_df.loc[mask_a, "Against"]
    data_df.loc[mask_a, "No-Match"] = (data_df.loc[mask_a, "In-Favor"] + data_df.loc[mask_a, "Neutral"] + data_df.loc[mask_a, "Error"])
    ##
    return data_df[list(set(data_df.columns)-set(["In-Favor", "Against", "Neutral", "Error"]))]


def output_latex(results, by_topic, by_prompt, match_by_topic, match_by_prompt):
    keys = ["In-Favor", "Against", "Neutral", "Error"]
    ##
    o = ""
    o += "% Basic and Extended Stances (Whole response)\n\n"
    for gid,gdf in results.groupby("Prompt"):
        o += gdf.set_index(["Model", "Tgt.Stance"])[keys].to_latex(caption=gid, float_format="{:.2f}".format)+"\n\n\n"
    # o += "% Basic and Extended Stances (Whole response) by topic\n\n"
    # o += by_topic.set_index(["Model", "Prompt", "Topic", "Tgt.Stance"])[keys].to_latex(float_format="{:.2f}".format)+"\n\n\n"
    # o += "% Basic and Extended Stances (Whole response) by prompt\n\n"
    # o += by_prompt.set_index(["Model", "Prompt", "Prompt-Lbl.", "Tgt.Stance"])[keys].to_latex(float_format="{:.2f}".format)+"\n\n\n"
    #############
    o += "% Matches (Whole response) by topic\n\n"
    for gid,gdf in match_by_topic.groupby("Prompt"):
        o += gdf.set_index(["Model", "Topic", "Tgt.Stance"])[["Match", "No-Match"]].to_latex(caption=gid, float_format="{:.2f}".format)+"\n\n\n"
    o += "% Matches (Whole response) by prompt\n\n"
    for gid,gdf in match_by_prompt.groupby("Prompt"):
        o += gdf.set_index(["Model", "Prompt-Lbl.", "Tgt.Stance"])[["Match", "No-Match"]].to_latex(caption=gid, float_format="{:.2f}".format)+"\n\n\n"
    return o

def make_plot_by_topic(df, model_name, ax):
    data = df[df["Model"]==model_name].drop(columns=["Model", "No-Match"])#.set_index(["Prompt", "Topic", "Tgt.Stance"]).plot(kind="bar")
    data = data.set_index(["Topic", "Prompt", "Tgt.Stance"]).unstack(level=1)
    data.columns = ["Basic", "Extended"]
    data.plot.bar(rot=90, ax=ax)
    ax.plot([-0.125, 0.875], [data.at[("EU-Exit", "against"), "Basic"], data.at[("EU-Exit", "infavor"), "Basic"]], marker="x", c="r", ls="--")
    ax.plot([-0.125+0.25, 0.875+0.25], [data.at[("EU-Exit", "against"), "Extended"], data.at[("EU-Exit", "infavor"), "Extended"]], marker="x", c="r", ls="--")
    #
    ax.plot([-0.125+2.0, 0.875+2.0], [data.at[("Immigration", "against"), "Basic"], data.at[("Immigration", "infavor"), "Basic"]], marker="x", c="r", ls="--")
    ax.plot([-0.125+2.25, 0.875+2.25], [data.at[("Immigration", "against"), "Extended"], data.at[("Immigration", "infavor"), "Extended"]], marker="x", c="r", ls="--")
    #
    ax.plot([-0.125+4.0, 0.875+4.0], [data.at[("Social-Equality", "against"), "Basic"], data.at[("Social-Equality", "infavor"), "Basic"]], marker="x", c="r", ls="--")
    ax.plot([-0.125+4.25, 0.875+4.25], [data.at[("Social-Equality", "against"), "Extended"], data.at[("Social-Equality", "infavor"), "Extended"]], marker="x", c="r", ls="--")
    
    #
    ax.set_xlabel("")

def make_plot_by_prompt(df, model_name, ax):
    data = df[df["Model"]==model_name].drop(columns=["Model", "No-Match"])#.set_index(["Prompt", "Topic", "Tgt.Stance"]).plot(kind="bar")
    data = data.set_index(["Prompt-Lbl.", "Prompt", "Tgt.Stance"]).unstack(level=1)
    data.columns = ["Basic", "Extended"]
    data.plot.bar(rot=90, ax=ax)
    ax.plot([-0.125, 0.875], [data.at[("Against", "against"), "Basic"], data.at[("Against", "infavor"), "Basic"]], marker="x", c="r", ls="--")
    ax.plot([-0.125+0.25, 0.875+0.25], [data.at[("Against", "against"), "Extended"], data.at[("Against", "infavor"), "Extended"]], marker="x", c="r", ls="--")
    #
    ax.plot([-0.125+2.0, 0.875+2.0], [data.at[("In-Favor", "against"), "Basic"], data.at[("In-Favor", "infavor"), "Basic"]], marker="x", c="r", ls="--")
    ax.plot([-0.125+2.25, 0.875+2.25], [data.at[("In-Favor", "against"), "Extended"], data.at[("In-Favor", "infavor"), "Extended"]], marker="x", c="r", ls="--")
    #
    ax.plot([-0.125+4.0, 0.875+4.0], [data.at[("Information", "against"), "Basic"], data.at[("Information", "infavor"), "Basic"]], marker="x", c="r", ls="--")
    ax.plot([-0.125+4.25, 0.875+4.25], [data.at[("Information", "against"), "Extended"], data.at[("Information", "infavor"), "Extended"]], marker="x", c="r", ls="--")
    #
    ax.plot([-0.125+6.0, 0.875+6.0], [data.at[("Neutral", "against"), "Basic"], data.at[("Neutral", "infavor"), "Basic"]], marker="x", c="r", ls="--")
    ax.plot([-0.125+6.25, 0.875+6.25], [data.at[("Neutral", "against"), "Extended"], data.at[("Neutral", "infavor"), "Extended"]], marker="x", c="r", ls="--")
    
    #
    ax.set_xlabel("")


if __name__ == "__main__":
    ## Count percentages
    gpt_res = evaluate_by_model("gpt_3_5_turbo")
    gpt_by_topic = evaluate_by_model_and("gpt_3_5_turbo", "topic")
    gpt_by_prompt = evaluate_by_model_and("gpt_3_5_turbo", "Prompt_Label")
    #
    mistral_res = evaluate_by_model("mistral_small")
    mistral_by_topic = evaluate_by_model_and("mistral_small", "topic")
    mistral_by_prompt = evaluate_by_model_and("mistral_small", "Prompt_Label")
    #
    gemma_res = evaluate_by_model("gemma3_4b")
    gemma_by_topic = evaluate_by_model_and("gemma3_4b", "topic")
    gemma_by_prompt = evaluate_by_model_and("gemma3_4b", "Prompt_Label")

    ## Rename columns and index
    gpt_res, gpt_by_topic, gpt_by_prompt = post_process_tables(gpt_res, gpt_by_topic, gpt_by_prompt, "GPT-3.5-turbo")
    mistral_res, mistral_by_topic, mistral_by_prompt = post_process_tables(mistral_res, mistral_by_topic, mistral_by_prompt, "Mistral-Small")
    gemma_res, gemma_by_topic, gemma_by_prompt = post_process_tables(gemma_res, gemma_by_topic, gemma_by_prompt, "Gemma3-4b")
    
    ## Concat models into one table
    results = pd.concat((gpt_res, mistral_res, gemma_res)).fillna(0)
    results_by_topic = pd.concat((gpt_by_topic, mistral_by_topic, gemma_by_topic)).fillna(0)
    results_by_prompt = pd.concat((gpt_by_prompt, mistral_by_prompt, gemma_by_prompt)).fillna(0)
    
    ##
    matches_by_topic = make_match_table(results_by_topic)
    matches_by_prompt = make_match_table(results_by_prompt)

    # ## 
    # test_df = matches_by_topic.set_index(["Model", "Topic", "Tgt.Stance"])
    # basic_df = test_df[test_df["Prompt"]=="Basic"].drop(columns="Prompt")
    # extended_df = test_df[test_df["Prompt"]=="Extended"].drop(columns="Prompt")
    # basic_df.columns = [["Basic", "Basic"], basic_df.columns]
    # extended_df.columns = [["Extended", "Extended"], extended_df.columns]
    # full_df = basic_df.join(extended_df)[[("Basic", "Match"), ("Basic", "No-Match"), ("Extended", "Match"), ("Extended", "No-Match")]]
    # print(full_df.to_latex(caption="By Topic", float_format="{:.2f}".format)+"\n\n\n")
    # ##
    # test_df = matches_by_prompt.set_index(["Model", "Prompt-Lbl.", "Tgt.Stance"])
    # basic_df = test_df[test_df["Prompt"]=="Basic"].drop(columns="Prompt")
    # extended_df = test_df[test_df["Prompt"]=="Extended"].drop(columns="Prompt")
    # basic_df.columns = [["Basic", "Basic"], basic_df.columns]
    # extended_df.columns = [["Extended", "Extended"], extended_df.columns]
    # full_df = basic_df.join(extended_df)[[("Basic", "Match"), ("Basic", "No-Match"), ("Extended", "Match"), ("Extended", "No-Match")]]
    # print(full_df.to_latex(caption="By Prompt", float_format="{:.2f}".format)+"\n\n\n")
    # exit()
    ##
    o = output_latex(results, results_by_topic, results_by_prompt, matches_by_topic, matches_by_prompt).replace("_", r"\_")
    print(o)
    with open("results/step2_stance_detection.tex", "w") as ofile:
        ofile.write(o)
    
    ##
    ax1 = plt.subplot(221)
    ax1.set_title("GPT-3.5-turbo")
    make_plot_by_topic(matches_by_topic, "GPT-3.5-turbo", ax1)
    ax2 = plt.subplot(222, sharex=ax1)
    ax2.set_title("Mistral-Small")
    make_plot_by_topic(matches_by_topic, "Mistral-Small", ax2)
    ax3 = plt.subplot(223)
    ax3.set_title("Gemma3-4b")
    make_plot_by_topic(matches_by_topic, "Gemma3-4b", ax3)

    plt.savefig("./results/StanceResults_by_Topic.png", bbox_inches='tight')
    plt.show()

    ##
    ax1 = plt.subplot(221)
    ax1.set_title("GPT-3.5-turbo")
    make_plot_by_prompt(matches_by_prompt, "GPT-3.5-turbo", ax1)
    ax2 = plt.subplot(222, sharex=ax1)
    ax2.set_title("Mistral-Small")
    make_plot_by_prompt(matches_by_prompt, "Mistral-Small", ax2)
    ax3 = plt.subplot(223)
    ax3.set_title("Gemma3-4b")
    make_plot_by_prompt(matches_by_prompt, "Gemma3-4b", ax3)

    plt.savefig("./results/StanceResults_by_Prompt.png", bbox_inches='tight')
    plt.show()