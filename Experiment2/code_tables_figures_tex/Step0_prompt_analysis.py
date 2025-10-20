import pandas as pd

from data import pd_read_jsonl



if __name__ == "__main__":
    ## Load user message data (with stance-labels)
    prompts_df = pd_read_jsonl("data/data_cleaned.PromptStances.SpecialSystemPrompt.jsonl")

    ##
    topic_counts = pd.DataFrame(prompts_df.groupby("topic").size()).rename(columns={0:"Count"})
    label_counts =  pd.DataFrame(prompts_df.groupby("Prompt_Label").size()).rename(columns={0:"Count"})
    print(topic_counts)
    print(label_counts)
