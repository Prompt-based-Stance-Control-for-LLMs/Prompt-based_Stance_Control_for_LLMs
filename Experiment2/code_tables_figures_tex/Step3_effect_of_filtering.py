import pandas as pd

from data import pd_read_jsonl
from evaluation import eval_paragraph,eval_whole


STANCE_DE2EN = {"Zustimmung":"In-Favor", "Ablehnung":"Against", "Neutral":"Neutral", None:"Error", "Information":"Information"}

def evaluate_pargraphs(basic_df, extended_df):
    new_table = list()
    for gid,gdf in basic_df.groupby("infavor_response_GlobalLabel"):
        basic_infavor = eval_paragraph(gdf["infavor_response_ParagraphsLabeled"])
        new_table.append(
            dict([("Prompt", "Basic"), ("Tgt.Stance","infavor"), ("Whole-Response-Label", STANCE_DE2EN[gid])]+list(basic_infavor.items()))
        )
    for gid,gdf in basic_df.groupby("against_response_GlobalLabel"):
        basic_against  = eval_paragraph(gdf["against_response_ParagraphsLabeled"])
        new_table.append(
            dict([("Prompt", "Basic"), ("Tgt.Stance","against"), ("Whole-Response-Label", STANCE_DE2EN[gid])]+list(basic_against.items()))
        )
    #
    for gid,gdf in extended_df.groupby("infavor_response_GlobalLabel"):
        extended_infavor = eval_paragraph(gdf["infavor_response_ParagraphsLabeled"])
        new_table.append(
            dict([("Prompt", "Extended"), ("Tgt.Stance","infavor"), ("Whole-Response-Label", STANCE_DE2EN[gid])]+list(extended_infavor.items()))
        )
    for gid,gdf in extended_df.groupby("against_response_GlobalLabel"):
        extended_against = eval_paragraph(gdf["against_response_ParagraphsLabeled"])
        new_table.append(
            dict([("Prompt", "Extended"), ("Tgt.Stance","against"), ("Whole-Response-Label", STANCE_DE2EN[gid])]+list(extended_against.items()))
        )
    new_table = pd.DataFrame(new_table).fillna(0)
    new_table = new_table.rename(columns=STANCE_DE2EN)
    ##
    return new_table

def evaluate_by_model_by_paragraph(model_name):
    ## Load and prepare data
    basic_df = pd_read_jsonl("evaluation_labels/responses_basic.data_cleaned."+model_name+".evaluated.jsonl")
    basic_filtered_df = pd_read_jsonl("evaluation_labels/responses_basic.data_cleaned."+model_name+".filtered.evaluated.jsonl")
    #
    extended_df = pd_read_jsonl("evaluation_labels/responses_noNumbering.data_cleaned."+model_name+".evaluated.jsonl")
    extended_filtered_df = pd_read_jsonl("evaluation_labels/responses_noNumbering.data_cleaned."+model_name+".filtered.evaluated.jsonl")
    
    ## Count stance percentages (paragraphs by whole)
    befor_filtering = evaluate_pargraphs(basic_df, extended_df)
    after_filtering = evaluate_pargraphs(basic_filtered_df, extended_filtered_df)
    ##
    return befor_filtering,after_filtering


def evaluate_whole(basic_df, extended_df):
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

def evaluate_by_model(model_name):
    ## Load and prepare data
    basic_df = pd_read_jsonl("evaluation_labels/responses_basic.data_cleaned."+model_name+".evaluated.jsonl")
    basic_filtered_df = pd_read_jsonl("evaluation_labels/responses_basic.data_cleaned."+model_name+".filtered.evaluated.jsonl")
    #
    extended_df = pd_read_jsonl("evaluation_labels/responses_noNumbering.data_cleaned."+model_name+".evaluated.jsonl")
    extended_filtered_df = pd_read_jsonl("evaluation_labels/responses_noNumbering.data_cleaned."+model_name+".filtered.evaluated.jsonl")
    
    ##
    befor = evaluate_whole(basic_df, extended_df)
    after = evaluate_whole(basic_filtered_df, extended_filtered_df)
    ##
    return befor,after


def count_filtered_paragraphs(model_name):
    ## Load and prepare data
    basic_df = pd_read_jsonl("responses_basic/data_cleaned."+model_name+".jsonl")
    basic_filtered_df = pd_read_jsonl("responses_basic/data_cleaned."+model_name+".filtered.jsonl")
    #
    extended_df = pd_read_jsonl("responses_noNumbering/data_cleaned."+model_name+".jsonl")
    extended_filtered_df = pd_read_jsonl("responses_noNumbering/data_cleaned."+model_name+".filtered.jsonl")
    #

    basic_filtered_df["infavor_removed"] = basic_filtered_df["infavor_removed_paragraphs"].apply(len)
    basic_filtered_df["against_removed"] = basic_filtered_df["against_removed_paragraphs"].apply(len)
    extended_filtered_df["infavor_removed"] = extended_filtered_df["infavor_removed_paragraphs"].apply(len)
    extended_filtered_df["against_removed"] = extended_filtered_df["against_removed_paragraphs"].apply(len)
    
    #
    num_removed_paragraphs = pd.DataFrame({
        "Basic-Infavor":basic_filtered_df.groupby("infavor_removed").size(),
        "Basic-Against":basic_filtered_df.groupby("against_removed").size(),
        "Extended-Infavor":extended_filtered_df.groupby("infavor_removed").size(),
        "Extended-Against":extended_filtered_df.groupby("against_removed").size()
    }).fillna(0)
    print(num_removed_paragraphs)

    ##
    debug1_i = len(basic_df[basic_df["infavor_response"].apply(len)==0])
    debug1_a = len(basic_df[basic_df["against_response"].apply(len)==0])
    debug2_i = len(extended_df[extended_df["infavor_response"].apply(len)==0])
    debug2_a = len(extended_df[extended_df["against_response"].apply(len)==0])
    #
    basic_infavor_num_empty = len(basic_filtered_df[basic_filtered_df["infavor_response"].apply(len)==0])
    basic_against_num_empty = len(basic_filtered_df[basic_filtered_df["against_response"].apply(len)==0])
    #
    extended_infavor_num_empty = len(extended_filtered_df[extended_filtered_df["infavor_response"].apply(len)==0])
    extended_against_num_empty = len(extended_filtered_df[extended_filtered_df["against_response"].apply(len)==0])
    print("\n\n")
    print("Befor filtering | Zero texts: ", debug1_a, debug1_i, debug2_a, debug2_i)
    print("Basic     |  Texts filtered down to Zero:", basic_infavor_num_empty, basic_against_num_empty)
    print("Extended  |  Texts filtered down to Zero:",extended_infavor_num_empty, extended_against_num_empty)



def merge_models(gpt, mistral, gemma):
    gpt["Model"] = "GPT-3.5-turbo"
    mistral["Model"] = "Mistral-Small"
    gemma["Model"] = "Gemma3-4b"
    #
    full_df = pd.concat((gpt,mistral,gemma)).fillna(0)
    full_df = full_df.rename(columns=STANCE_DE2EN)
    ##
    return full_df

def output_latex(parag_befor, whole_delta, parag_delta, parag_after):
    keys_para = ["In-Favor", "Against", "Neutral"]  # , "Error"
    keys_whole = ["In-Favor", "Against", "Neutral", "Error"]  # 
    o = ""

    o +="% Paragraphs by whole-response (BEFOR FILTERING)\n"
    for gid,gdf in parag_befor.groupby("Prompt"):
        gdf = gdf.set_index(["Model", "Tgt.Stance", "Whole-Response-Label"])
        ##
        cat1 = pd.CategoricalIndex(gdf.index.levels[1].values,
                                   categories=["infavor", "against"],
                                   ordered=True)
        cat2 = pd.CategoricalIndex(gdf.index.levels[2].values,
                                   categories=["In-Favor", "Neutral", "Against"],
                                   ordered=True)
        gdf.index = gdf.index.set_levels(cat1, level=1)
        gdf.index = gdf.index.set_levels(cat2, level=2)
        gdf = gdf.sort_index()
        ##
        o += gdf[keys_para].to_latex(caption=gid, float_format="{:.2f}".format)+"\n\n\n"

    o +="% Whole responses: Delta due to filtering\n"
    print(whole_delta)
    for gid,gdf in whole_delta.groupby("Prompt"):
        o += gdf.set_index(["Model", "Tgt.Stance",])[keys_whole].to_latex(caption=gid, float_format="{:.2f}".format)+"\n\n\n"
    
    o +="% Paragraphs by whole-response: Delta due to filtering\n"
    for gid,gdf in parag_delta.groupby("Prompt"):
        gdf = gdf.set_index(["Model", "Tgt.Stance", "Whole-Response-Label"])
        ##
        cat1 = pd.CategoricalIndex(gdf.index.levels[1].values,
                                   categories=["infavor", "against"],
                                   ordered=True)
        cat2 = pd.CategoricalIndex(gdf.index.levels[2].values,
                                   categories=["In-Favor", "Neutral", "Against"],
                                   ordered=True)
        gdf.index = gdf.index.set_levels(cat1, level=1)
        gdf.index = gdf.index.set_levels(cat2, level=2)
        gdf = gdf.sort_index()
        ##
        o += gdf[keys_para].to_latex(caption=gid, float_format="{:.2f}".format)+"\n\n\n"
    
    o +="% Paragraphs by whole-response (AFTER FILTERING)\n"
    for gid,gdf in parag_after.groupby("Prompt"):
        gdf = gdf.set_index(["Model", "Tgt.Stance", "Whole-Response-Label"])
        ####
        cat1 = pd.CategoricalIndex(gdf.index.levels[1].values,
                                   categories=["infavor", "against"],
                                   ordered=True)
        cat2 = pd.CategoricalIndex(gdf.index.levels[2].values,
                                   categories=["In-Favor", "Neutral", "Against"],
                                   ordered=True)
        gdf.index = gdf.index.set_levels(cat1, level=1)
        gdf.index = gdf.index.set_levels(cat2, level=2)
        gdf = gdf.sort_index()
        ##
        o += gdf[keys_para].to_latex(caption=gid, float_format="{:.2f}".format)+"\n\n\n"


    return o


if __name__ == "__main__":
    #count_filtered_paragraphs("gpt_3_5_turbo")

    ## Count percentages
    gpt_parag_befor, gpt_parag_after = evaluate_by_model_by_paragraph("gpt_3_5_turbo")
    gpt_whole_befor, gpt_whole_after = evaluate_by_model("gpt_3_5_turbo")
    gpt_whole_delta = gpt_whole_after.set_index(["Prompt", "Tgt.Stance"])-gpt_whole_befor.set_index(["Prompt", "Tgt.Stance"])
    gpt_parag_delta = gpt_parag_after.set_index(["Prompt", "Tgt.Stance", "Whole-Response-Label"])-gpt_parag_befor.set_index(["Prompt", "Tgt.Stance", "Whole-Response-Label"])
    #
    mistral_parag_befor, mistral_parag_after = evaluate_by_model_by_paragraph("mistral_small")
    mistral_whole_befor, mistral_whole_after = evaluate_by_model("mistral_small")
    mistral_whole_delta = mistral_whole_after.set_index(["Prompt", "Tgt.Stance"])-mistral_whole_befor.set_index(["Prompt", "Tgt.Stance"])
    mistral_parag_delta = mistral_parag_after.set_index(["Prompt", "Tgt.Stance", "Whole-Response-Label"])-mistral_parag_befor.set_index(["Prompt", "Tgt.Stance", "Whole-Response-Label"])
    #
    gemma_parag_befor, gemma_parag_after = evaluate_by_model_by_paragraph("gemma3_4b")
    gemma_whole_befor, gemma_whole_after = evaluate_by_model("gemma3_4b")
    gemma_whole_delta = gemma_whole_after.set_index(["Prompt", "Tgt.Stance"])-gemma_whole_befor.set_index(["Prompt", "Tgt.Stance"])
    gemma_parag_delta = gemma_parag_after.set_index(["Prompt", "Tgt.Stance", "Whole-Response-Label"])-gemma_parag_befor.set_index(["Prompt", "Tgt.Stance", "Whole-Response-Label"])
    
    ##
    whole_befor = merge_models(gpt_whole_befor,mistral_whole_befor,gemma_whole_befor)
    whole_after = merge_models(gpt_whole_after,mistral_whole_after,gemma_whole_after)
    whole_delta = merge_models(gpt_whole_delta,mistral_whole_delta,gemma_whole_delta).reset_index()
    #
    parag_befor = merge_models(gpt_parag_befor,mistral_parag_befor,gemma_parag_befor)
    parag_after = merge_models(gpt_parag_after,mistral_parag_after,gemma_parag_after)
    parag_delta = merge_models(gpt_parag_delta,mistral_parag_delta,gemma_parag_delta).reset_index()
    #.reset_index().set_index(["Model", "Prompt", "Tgt.Stance", "Whole-Response-Label"])


    #
    o = output_latex(parag_befor, whole_delta, parag_delta, parag_after).replace("_", r"\_")
    #  print(o)
    with open("results/step3_paragraphs_and_filtering.tex", "w") as ofile:
        ofile.write(o)