# Stance Detection 
This part of the repository contains all scripts used to evaluate LLM-based stance detection methods.

## Repository structure and file descriptions
In the `code` folder, all scripts can be found, used for the first part of the paper.
In the folders prefixed with `exp_...` contain all predictions and artefacts associated with each tested model/prompt.
For the LLM-based approaches this includes the prompt used for stance detection.

| File | Description |
| -- | -- |
| ./code/M1_fine_tune_lm.py | Fine-tune a BERT-based classifier.  |
| ./code/M1_predict_lm.py | Predict a dataset with a fine-tuned BERT-based classifier. |
| ./code/M2_predict_llm_ollama.py | Stance detection with an LLM from Ollama. (LLM-based method) |
| ./code/M3_prepare_for_bert_with_llm.py | Rephrases the texts from the x-stance dataset, befor fine-tuning a Bert-based classifier (LLM-prepared-Bert method) |
| ./code/M4_combine_bert_and_llm.py | Combines the predictions from the LLM-based method with the GermanBert predictions (Bert-then-LLM method) |
| ./code/M5_combine_into_voting.py | Combines existing predictions into a voted prediction (voting method)  |
| ./code/evaluate_predictions.py | Script used to evaluate all the produced predictions against a gold file |
| ./run_exp_LLM-based.sh | Script used to run LLM-based stance detection with all prompt versions and models |
| ./run_eval_LLM-based.sh | Script used to evaluate results against the gold data |
| ./other_approaches_example.sh | Script explaining how to run the BERT-based and the voting approach |

All the above scripts (except of M5_combine_into_voting.py) can be used with "python3 <script_name> --help".

The LLM-based experiments were carried out, using the sh-scripts found in the base folder.
The GermanBert baseline, LLM-prepared-Bert, Bert-then-LLM, and the Voting approach were manually started.
To see how these other approaches could be re-created, see the other_approaches_example.sh script.

## Usage
### Setup
 - Download the [xstance](https://github.com/ZurichNLP/xstance) data (three files: train.jsonl, valid.jsonl, and test.jsonl)
 - Create a python3.11 virtual environment and install the requirements.txt
 - Setup ollama on your system. See [here](https://github.com/ollama/ollama) for installation process.
 - Befor running the experiments, pull all needed LLMs with ollama. For a list of used models see the run_exp_LLM-based.sh script.

### Reproduction of experiments
- For the LLM-based approach, simply run the sh scripts; Please manually adjust the data variable in the scripts to eather valid.jsonl or test.jsonl (similar in both scripts).
- For the other approaches, see the example sh-script. Here you must manually set path to your local files.
- For the voting approach, set the paths in the M5_combine_into_voting.py to your local files and run it. Paths are hard coded.

