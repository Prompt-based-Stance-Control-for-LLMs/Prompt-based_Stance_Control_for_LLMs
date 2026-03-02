from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import langdetect 
from tqdm import tqdm
import pandas as pd
import re
import sys

from data import pd_read_jsonl, write_jsonl



model = OllamaLLM(model="mistral-small",)
def query_llm(inputs, system, labels=["ja", "nein"]):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Eingabe: '{prompt}'")
    ])
    chain = prompt|model
    #
    outputs = []
    for x in tqdm(inputs, total=len(inputs)):
        res = ""
        print("+"*25)
        print(x)
        for _ in range(5):
            res = chain.invoke({"prompt": x}).lower()
            print("="*25)
            print(res)
            if res in labels:
                break
        print("+"*25)
        outputs.append(res)

    return outputs


if __name__=="__main__":
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    print(in_path+"\n"+out_path,"\n\n")
    data_raw = pd_read_jsonl(in_path)

    data_mig     = data_raw[data_raw["topic"]=="immigration"].reset_index(drop=True)
    data_eu_exit = data_raw[data_raw["topic"]=="EU_exit"].reset_index(drop=True)
    data_sequ    = data_raw[data_raw["topic"]=="social_equality"].reset_index(drop=True)

    data_mig["llm_migration?"] = query_llm(data_mig["prompt"], """
Enthält der Text einen Bezug zum Thema "Migration (Immigration, Zuwanderung, Einwanderung, Flüchtlinge, Asylanten) nach Deutschland (BRD)"?
Antworte nur mit Ja oder Nein
""")
    
    data_eu_exit["llm_eu_exit?"] = query_llm(data_eu_exit["prompt"], """
Enthält der Text einen Bezug zum Thema "Deutschlands austritt aus der EU (die EU verlassen)" oder "Deutschland als EU Mitglied"?
Antworte nur mit Ja oder Nein
""")
        
    data_sequ["llm_social_equality?"] = query_llm(data_sequ["prompt"], """
Enthält der Text einen Bezug zum Thema "Soziale Gleichheit, Gerechtigkeit, Ungleichheit"?
Antworte nur mit Ja oder Nein
""")    


    data_mig["llm_migration?"] = data_mig["llm_migration?"].apply(lambda x:x=="ja")
    data_eu_exit["llm_eu_exit?"] = data_eu_exit["llm_eu_exit?"].apply(lambda x:x=="ja")
    data_sequ["llm_social_equality?"] = data_sequ["llm_social_equality?"].apply(lambda x:x=="ja")

    data_prep = pd.concat([data_mig, data_eu_exit, data_sequ])
    
    data_prep["question?"] = query_llm(data_prep["prompt"], """
    Enthält der Text eine Frage, eine Anweisung oder eine Aussage?
    Manche fragen sind sehr kurz formuliert und ohne Fragezeichen trozdem können es Fragen sein.
    Wenn der Text keine Frage, Anweisung oder Aussage enthält, anworte mit Sonstiges.
    Anworte nur mit Frage, Aussage, Anweisung oder Sonstiges.
                                        
    ##Beispiel
    Text: Vor und nachteile von Bananen
    Ausgabe: Anweisung

    ##Beispiel
    Text: Was sind Bananen?
    Ausgabe: Frage

    ##Beispiel
    Text: Bananen sind gelb und gebogen
    Ausgabe: Aussage
    """, labels=["frage", "aussage", "anweisung", "sonstiges"])    

    ##
    data_prep["num. tokens"] = data_prep["prompt"].apply(lambda x:len(re.sub(r"\s|\n", " ", x).split(" ")))

    data_prep.to_excel(out_path+".xlsx", index=False)