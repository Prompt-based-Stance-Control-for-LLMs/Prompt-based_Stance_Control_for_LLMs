# Stance Control
This part of the repository contains all the scripts used to evaluate stance control methods for LLMs.

## Files and repo structure
This part of the repository is structured as follows: 
The code used to produce and evaluate LLM responses with stance detection is located in the `code` folder.

Folders prefixed with `responses_` contain a text file with the respective system prompt, used to generate the controlled responses. The generated responses are also stored there as JSONL files.

The `evaluation_labels` folder contains a text file with the prompt used for stance detection, as well as the evaluation labels (stance labels) for all responses.

The `code_tables_figures_text` folder contains the code used to produce the tables and figures in the paper. Please note that this code is not abstracted or generalised.


| Path | Description |
|--|--|
| code/E_stance_detection.py | Script used to predict the stance labels for a given JSONL file containing the responses. |
| code/E_stance_detection_PROMPTS.py | Script used to predict the stance labels for the user messages from the PolPrompts dataset. |
| code/M0_....py | Scripts used for data cleaning. One script uses an LLM to predict whether a user's message is relevant to the topic and what type of request it is. The other splits the dataset into cleaned and removed parts.|
| code/M1_generate_vanilla.py | Feeds the user messages from the specified JSONL file into an uncontrolled LLM |
| code/M2_generate_controlled.py | Feeds the user messages from the specified JSONL file into an LLM, once with the infavor.txt system prompt and once with the against.txt system prompt, both of which are located in the specified working directory.  |
| code/M3_filter_responses.py | Take a file containing LLM responses and filter out any paragraphs that do not match the target stance. Use the specified prompt for stance detection. |
| ./run_experiment_<model_name>.sh | Script used to perform all the experiments for each model. See the script for examples of the arguments for all the above-mentioned scripts.  |
| ./topic_descriptions.json | This contains the natural language descriptions of the topics used to fill the prompt templates. Specify this as an argument in the above scripts. | 


## Usage
### Setup
 - Prepare your data such that it has two fields *topic* and *prompt*. Where *topic* contains a key from the `topic_descriptions.json` file and *prompt* contains a message to which the LLM should respond to.
 - Create a python3.11 virtual environment and install the requirements.txt
 - Setup ollama on your system. See [here](https://github.com/ollama/ollama) for installation process.
 - Befor running the experiments pull all needed LLMs with ollama, eg. `qwen2.5:14b`, `gemma3:4`, `mistral-small`
 - For the `GPT3.5-turbo` model, create an API-key for your OpenAI subscription

### Reproduction of experiments
 - Place the input data in the data folder (call it data_cleaned.jsonl or adjust the data variable in the sh script).
 - Then run the sh scripts.
 - If the GPT script is used, fist export the environment variabel `OPENAI_API_KEY` filled with your openAI subscription key.
 - In case you are unsure about the arguments of the above mentioned scripts use `python3 <script_name> --help`.
