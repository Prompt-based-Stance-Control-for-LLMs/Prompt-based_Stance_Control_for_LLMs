# Stance Detection 
This part of the repository contains all scripts used to evaluate LLM-based stance detection methods.

## Repository structure and file descriptions
In the `code` folder, all scripts can be found, used for the first part of the paper.
Folders prefixed with `exp_...` contain all predictions and artefacts associated with each tested model/prompt.
For the LLM-based approaches this includes the prompt used for stance detection.

| File | Description |
| -- | -- |
| ./code/M1_fine_tune_lm.py | Fine-tune a BERT-based classifier.  |
| ./code/M1_predict_lm.py | Use a fine-tuned BERT-based classifier to predict a dataset.|
| ./code/M2_predict_llm_ollama.py | Stance detection using an LLM from Ollama. |
| ./code/M3_prepare_for_bert_with_llm.py | Rephrases texts from the X-Stance dataset, befor fine-tuning a BERT-based classifier |
| ./code/M4_combine_bert_and_llm.py | Combines the predictions from the LLM-based method with those from GermanBert |
| ./code/M5_combine_into_voting.py | Combines existing predictions into a voted prediction |
| ./code/evaluate_predictions.py | Code used to evaluate all the produced predictions
| ./run_exp_LLM-based.sh | Script used to run LLM-based stance detection with all prompt versions and models |
| ./run_eval_LLM-based.sh | Script used to evaluate results against the gold data |
| ./other_approaches_example.sh | Script explaining how to run the BERT-based and the voting approach |

All the above scripts (excluding `M5_combine_into_voting.py`) an be used with the command `python3 <script_name> --help`.

The LLM-based experiments were carried out, using the shell-scripts found in the base folder.
The GermanBert baseline, LLM-prepared-Bert, Bert-then-LLM, and the Voting  approaches were started manually.
To see how these other approaches could be recreated, refer to the  `other_approaches_example.sh` script.

## Usage
### Setup
 - Download the [xstance](https://github.com/ZurichNLP/xstance) data (three files: train.jsonl, valid.jsonl, and test.jsonl)
 - Create a python3.11 virtual environment and install the requirements.txt
 - Setup ollama on your system. See [here](https://github.com/ollama/ollama) for installation process.
 - Befor running the experiments, pull all needed LLMs with ollama. For a list of used models see the `run_exp_LLM-based.sh` script.

### Reproduction of experiments
- For the LLM-based approach, simply run the sh scripts; Please manually adjust the data variable in the scripts to  either  valid.jsonl or test.jsonl
- For the other approaches, see the example sh-script. Here you must manually set the path to your local files.
- For the voting approach, set the paths in the `M5_combine_into_voting.py` file to your local paths and run it. Paths are hard coded.

