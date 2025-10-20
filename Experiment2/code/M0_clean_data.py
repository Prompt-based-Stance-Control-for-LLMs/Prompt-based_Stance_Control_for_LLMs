import pandas as pd

from data import write_jsonl




data = pd.read_excel("data_cleaning/full_data.data_cleaning.V3.numTokens.MANUALY_ANNOTED.xlsx")
print(data)

kept_part   = data[data["REMOVE"]!=1]
droped_part = data[data["REMOVE"]==1]

print("Keeping {} rows".format(len(kept_part)))
print("Dropping {} rows".format(len(droped_part)))

droped_part.to_excel("data_cleaning/cleaned_data.removed_part.xlsx")
write_jsonl([e.to_dict() for _,e in kept_part.iterrows()], "data/data_cleaned.jsonl")