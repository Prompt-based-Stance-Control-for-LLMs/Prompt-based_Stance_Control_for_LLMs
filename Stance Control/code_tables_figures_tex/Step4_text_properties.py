import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15) 
matplotlib.rc('legend', fontsize=16 )
matplotlib.rc('axes', titlesize=18, labelsize=18)
matplotlib.rc('figure', figsize=(12,10), dpi=180)
import pandas as pd
from somajo import SoMaJo
from tqdm import tqdm
import pickle
import re

from data import pd_read_jsonl


tokenizer = SoMaJo("de_CMC",)
def words_sentences_paragraph_counts(texts):
    output = {"nSentences":list(), "nWords":list(), "nURLs":list(), "nParagraphs":list()}
    for text in tqdm(texts):
        sentences = list(tokenizer.tokenize_text([text,]))
        #
        output["nSentences"].append(len(sentences))
        output["nWords"].append(sum([len(s) for s in sentences]))
        output["nURLs"].append(len([e for s in sentences for e in s if e.token_class=="URL"]))
        output["nParagraphs"].append(len(text.split("\n\n")))
    ##
    return pd.DataFrame(output)

def headlines_bolt_enumerations(texts):
    output = {"nHeadlines":list(), "nEnumeration":list(), "nBulletpoints":list(), "nBold":list()}
    for text in tqdm(texts):
        output["nHeadlines"].append(text.count("\n###"))
        output["nEnumeration"].append(len(re.findall(r"^\s*\d+\..+$", text, flags=re.M)))
        output["nBulletpoints"].append(len(re.findall(r"^\s*\-.+$", text, flags=re.M)))
        output["nBold"].append(len(re.findall(r"\*\*(.+?)\*\*", text, flags=re.M)))
    return pd.DataFrame(output)

def generate_text_properties(model_name):
    ## Load and prepare data
    vanilla_df = pd_read_jsonl("responses_vanilla/data_cleaned."+model_name+".jsonl")
    basic_df = pd_read_jsonl("responses_basic/data_cleaned."+model_name+".jsonl")
    extended_df = pd_read_jsonl("responses_noNumbering/data_cleaned."+model_name+".jsonl")
    ##
    print("Vanilla ...")
    vanilla_counts = words_sentences_paragraph_counts(vanilla_df["response"])
    vanilla_counts = vanilla_counts.join(headlines_bolt_enumerations(vanilla_df["response"]))
    vanilla_counts = vanilla_counts.join(vanilla_df[["response"]])
    ##
    print("Basic ...")
    basic_infavor_counts = words_sentences_paragraph_counts(basic_df["infavor_response"])
    basic_infavor_counts = basic_infavor_counts.join(headlines_bolt_enumerations(basic_df["infavor_response"]))
    basic_infavor_counts = basic_infavor_counts.join(basic_df[["infavor_response"]])
    #
    basic_against_counts = words_sentences_paragraph_counts(basic_df["against_response"])
    basic_against_counts = basic_against_counts.join(headlines_bolt_enumerations(basic_df["against_response"]))
    basic_against_counts = basic_against_counts.join(basic_df[["against_response"]])
    ##
    print("Extended ...")
    extended_infavor_counts = words_sentences_paragraph_counts(extended_df["infavor_response"])
    extended_infavor_counts = extended_infavor_counts.join(headlines_bolt_enumerations(extended_df["infavor_response"]))
    extended_infavor_counts = extended_infavor_counts.join(extended_df[["infavor_response"]])
    #
    extended_against_counts = words_sentences_paragraph_counts(extended_df["against_response"])
    extended_against_counts = extended_against_counts.join(headlines_bolt_enumerations(extended_df["against_response"]))
    extended_against_counts = extended_against_counts.join(extended_df[["against_response"]])

    return vanilla_counts,basic_infavor_counts,basic_against_counts,extended_infavor_counts,extended_against_counts



def stats_table(count_df, prompt_name, tgt_stance):
    count_df = count_df.drop(columns=[e for e in count_df.columns if "response" in e])
    stats_df =  pd.DataFrame({
        "Min":count_df.min(),
        "P05":count_df.quantile(0.05),
        "P25":count_df.quantile(0.25),
        "Median":count_df.median(),
        "Mean":count_df.mean(),
        "P75":count_df.quantile(0.75),
        "P95":count_df.quantile(0.95),
        "Max":count_df.max(),
    })
    stats_df.index = [[prompt_name,]*len(stats_df.index), [tgt_stance,]*len(stats_df.index), stats_df.index]
    stats_df.index.set_names(["Prompt", "Tgt. Stance", "Property"])
    return stats_df

def make_stats_table(count_data):
    return pd.concat((
        stats_table(count_data[0], "Vanilla", "-"),
        stats_table(count_data[1], "Basic", "infavor"),
        stats_table(count_data[2], "Basic", "against"),
        stats_table(count_data[3], "Extended", "infavor"),
        stats_table(count_data[4], "Extended", "against"),
    ))



def plot_hists(count_data, key, ax, bins=list(range(0,500,25))):
    vanilla,basic_infavor,basic_against,extended_infavor,extended_against = count_data
    vanilla[key].hist(bins=bins,          color="tab:green",   label="Vanilla",            alpha=0.6, ax=ax)
    basic_infavor[key].hist(bins=bins,    color="tab:blue",    label="Basic - Infavor",    alpha=0.6, ax=ax)
    basic_against[key].hist(bins=bins,    color="tab:orange",  label="Basic - Against",    alpha=0.6, ax=ax)
    extended_infavor[key].hist(bins=bins, color="tab:cyan",    label="Extended - Infavor", alpha=0.6, ax=ax)
    extended_against[key].hist(bins=bins, color="tab:red",     label="Extended - Against", alpha=0.6, ax=ax)


def plot_hists_all_models(gpt_counts, mistral_counts, gemma_counts, key, bins):
    ##
    ax1 = plt.subplot(221)
    plot_hists(gpt_counts, key, ax1, bins=bins)
    ax1.set_title("GPT-3.5-turbo")
    #
    ax2 = plt.subplot(222, sharex=ax1)
    plot_hists(mistral_counts, key, ax2, bins=bins)
    ax2.set_title("Mistral-Small")
    #
    ax3 = plt.subplot(223, sharex=ax2)
    plot_hists(gemma_counts, key, ax3, bins=bins)
    ax3.set_title("Gemma3-4b")
    #
    ax3.set_xlim((0, max(bins)))
    ax3.legend(bbox_to_anchor=(1.2, 0),
               loc='lower left', borderaxespad=0., prop={'size': 16})


def make_latex(gpt_stats_data, mistral_stats_data, gemma_stats_data):
    o = ""

    ## Words,Paragraphs,Sentences
    gpt_words_sente_paras = gpt_stats_data.loc[[
        ("Vanilla", "-", "nWords"),
        ("Vanilla", "-", "nSentences"),
        ("Vanilla", "-", "nParagraphs"),

        ("Basic", "infavor", "nWords"),
        ("Basic", "infavor", "nSentences"),
        ("Basic", "infavor", "nParagraphs"),

        ("Basic", "against", "nWords"),
        ("Basic", "against", "nSentences"),
        ("Basic", "against", "nParagraphs"),

        ("Extended", "infavor", "nWords"),
        ("Extended", "infavor", "nSentences"),
        ("Extended", "infavor", "nParagraphs"),

        ("Extended", "against", "nWords"),
        ("Extended", "against", "nSentences"),
        ("Extended", "against", "nParagraphs")
    ]]
    mistral_words_sente_paras = mistral_stats_data.loc[[
        ("Vanilla", "-", "nWords"),
        ("Vanilla", "-", "nSentences"),
        ("Vanilla", "-", "nParagraphs"),

        ("Basic", "infavor", "nWords"),
        ("Basic", "infavor", "nSentences"),
        ("Basic", "infavor", "nParagraphs"),

        ("Basic", "against", "nWords"),
        ("Basic", "against", "nSentences"),
        ("Basic", "against", "nParagraphs"),

        ("Extended", "infavor", "nWords"),
        ("Extended", "infavor", "nSentences"),
        ("Extended", "infavor", "nParagraphs"),

        ("Extended", "against", "nWords"),
        ("Extended", "against", "nSentences"),
        ("Extended", "against", "nParagraphs")
    ]]
    gemma_words_sente_paras = gemma_stats_data.loc[[
        ("Vanilla", "-", "nWords"),
        ("Vanilla", "-", "nSentences"),
        ("Vanilla", "-", "nParagraphs"),

        ("Basic", "infavor", "nWords"),
        ("Basic", "infavor", "nSentences"),
        ("Basic", "infavor", "nParagraphs"),

        ("Basic", "against", "nWords"),
        ("Basic", "against", "nSentences"),
        ("Basic", "against", "nParagraphs"),

        ("Extended", "infavor", "nWords"),
        ("Extended", "infavor", "nSentences"),
        ("Extended", "infavor", "nParagraphs"),

        ("Extended", "against", "nWords"),
        ("Extended", "against", "nSentences"),
        ("Extended", "against", "nParagraphs")
    ]]
    #
    keys = ["Min", "P25", "Median", "Mean", "P75", "Max"]
    for k in gpt_words_sente_paras.columns:
        if k!="Mean":
            gpt_words_sente_paras[k] = gpt_words_sente_paras[k].apply(int)
            mistral_words_sente_paras[k] = mistral_words_sente_paras[k].apply(int)
            gemma_words_sente_paras[k] = gemma_words_sente_paras[k].apply(int)
    o += gpt_words_sente_paras[keys].to_latex(caption="GPT-3.5-turbo", float_format="{:.2f}".format)+"\n\n\n"
    o += mistral_words_sente_paras[keys].to_latex(caption="Mistral-Small", float_format="{:.2f}".format)+"\n\n\n"
    o += gemma_words_sente_paras[keys].to_latex(caption="Gemma3-4b", float_format="{:.2f}".format)+"\n\n\n"

    ## Words,Paragraphs,Sentences
    gpt_words_sente_paras = gpt_stats_data.loc[[
        ("Vanilla", "-", "nHeadlines"),
        ("Vanilla", "-", "nEnumeration"),
        ("Vanilla", "-", "nBold"),

        ("Basic", "infavor", "nHeadlines"),
        ("Basic", "infavor", "nEnumeration"),
        ("Basic", "infavor", "nBold"),

        ("Basic", "against", "nHeadlines"),
        ("Basic", "against", "nEnumeration"),
        ("Basic", "against", "nBold"),

        ("Extended", "infavor", "nHeadlines"),
        ("Extended", "infavor", "nEnumeration"),
        ("Extended", "infavor", "nBold"),

        ("Extended", "against", "nHeadlines"),
        ("Extended", "against", "nEnumeration"),
        ("Extended", "against", "nBold")
    ]]
    mistral_words_sente_paras = mistral_stats_data.loc[[
        ("Vanilla", "-", "nHeadlines"),
        ("Vanilla", "-", "nEnumeration"),
        ("Vanilla", "-", "nBold"),

        ("Basic", "infavor", "nHeadlines"),
        ("Basic", "infavor", "nEnumeration"),
        ("Basic", "infavor", "nBold"),

        ("Basic", "against", "nHeadlines"),
        ("Basic", "against", "nEnumeration"),
        ("Basic", "against", "nBold"),

        ("Extended", "infavor", "nHeadlines"),
        ("Extended", "infavor", "nEnumeration"),
        ("Extended", "infavor", "nBold"),

        ("Extended", "against", "nHeadlines"),
        ("Extended", "against", "nEnumeration"),
        ("Extended", "against", "nBold")
    ]]
    gemma_words_sente_paras = gemma_stats_data.loc[[
        ("Vanilla", "-", "nHeadlines"),
        ("Vanilla", "-", "nEnumeration"),
        ("Vanilla", "-", "nBold"),

        ("Basic", "infavor", "nHeadlines"),
        ("Basic", "infavor", "nEnumeration"),
        ("Basic", "infavor", "nBold"),

        ("Basic", "against", "nHeadlines"),
        ("Basic", "against", "nEnumeration"),
        ("Basic", "against", "nBold"),

        ("Extended", "infavor", "nHeadlines"),
        ("Extended", "infavor", "nEnumeration"),
        ("Extended", "infavor", "nBold"),

        ("Extended", "against", "nHeadlines"),
        ("Extended", "against", "nEnumeration"),
        ("Extended", "against", "nBold")
    ]]
    #
    keys = ["Min", "P25", "Median", "Mean", "P75", "Max"]
    for k in gpt_words_sente_paras.columns:
        if k!="Mean":
            gpt_words_sente_paras[k] = gpt_words_sente_paras[k].apply(int)
            mistral_words_sente_paras[k] = mistral_words_sente_paras[k].apply(int)
            gemma_words_sente_paras[k] = gemma_words_sente_paras[k].apply(int)
    o += gpt_words_sente_paras[keys].to_latex(caption="GPT-3.5-turbo", float_format="{:.2f}".format)+"\n\n\n"
    o += mistral_words_sente_paras[keys].to_latex(caption="Mistral-Small", float_format="{:.2f}".format)+"\n\n\n"
    o += gemma_words_sente_paras[keys].to_latex(caption="Gemma3-4b", float_format="{:.2f}".format)+"\n\n\n"


    ##
    return o

if __name__ == "__main__":
    ## Count percentages
    if False:
        gpt_counts = generate_text_properties("gpt_3_5_turbo")
        mistral_counts = generate_text_properties("mistral_small")
        gemma_counts = generate_text_properties("gemma3_4b")
        with open("text_prop_save.pickle", "bw") as ofile:
            pickle.dump({"GPT":gpt_counts, "Mistral":mistral_counts, "Gemma":gemma_counts}, ofile)
    else:
        with open("text_prop_save.pickle", "br") as ifile:
            data = pickle.load(ifile)
        gpt_counts = data["GPT"]
        mistral_counts = data["Mistral"]
        gemma_counts = data["Gemma"]
    ##
    gpt_stats_data = make_stats_table(gpt_counts)
    mistral_stats_data = make_stats_table(mistral_counts)
    gemma_stats_data = make_stats_table(gemma_counts)

    ##
    with open("results/step4_text_properties.text", "tw") as ofile:
        o = make_latex(gpt_stats_data, mistral_stats_data, gemma_stats_data)
        ofile.write(o)
    print(o)
    ##
    plot_hists_all_models(gpt_counts, mistral_counts, gemma_counts, "nWords", bins=list(range(1,751,25)))
    plt.savefig("results/textProps_nWords.png", bbox_inches='tight')
    plt.show()

    plot_hists_all_models(gpt_counts, mistral_counts, gemma_counts, "nParagraphs", bins=list(range(1,21,1)))
    plt.savefig("results/textProps_nParagraphs.png", bbox_inches='tight')
    plt.show()

    # plot_hists_all_models(gpt_counts, mistral_counts, gemma_counts, "nSentences", bins=list(range(1,31,1)))
    # plt.savefig("results/textProps_nSentences.png", bbox_inches='tight')
    # plt.show()


    plot_hists_all_models(gpt_counts, mistral_counts, gemma_counts, "nEnumeration", bins=list(range(1,21,1)))
    plt.savefig("results/textProps_nEnumeration.png", bbox_inches='tight')
    plt.show()

    plot_hists_all_models(gpt_counts, mistral_counts, gemma_counts, "nBold", bins=list(range(1,41,1)))
    plt.savefig("results/textProps_nBold.png", bbox_inches='tight')
    plt.show()

    plot_hists_all_models(gpt_counts, mistral_counts, gemma_counts, "nHeadlines", bins=list(range(1,41,1)))
    plt.savefig("results/textProps_nBold.png", bbox_inches='tight')
    plt.show()