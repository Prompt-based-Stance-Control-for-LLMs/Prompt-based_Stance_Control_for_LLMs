# Stance Control

### Files and Repo structure
This part of the repository is structed as follows:
The code used to produce the LLM responses, and to evaluate with stance detection is place in the code folder.

The folders prefixed with "responses_" contain a txt file with the respective system prompt, used to genreate the controlled responses. Also the generated responses will be stored there as jsonl files.

The evaluation_labels folder holds a txt file with the prompt used for stance detection and will hold the evaluation labels (stance labels) for all responses.

The code_tables_figures_text folder holds the code that produced the tables and figures for the thesis. Note that this code is not abstracted or generalized and contains local path that I didn't adjusted, for reproducing the tables one must manually adjust the hard coded paths.


| Path | Description |
|--|--|
| code/E_stance_detection.py | Script used to predict the stance labels for a given jsonl file containing the responses. |
| code/E_stance_detection_PROMPTS.py | Script used to predict the stance labels for the user messages from the UserMessage dataset. |
| code/M0_....py | Scripts used for data cleaning. One uses an LLM to predict whether the user message is relevant for the topic and what kind of request it is. The other splits the dataset into the cleaned and the removed part. |
| code/M1_generate_vanilla.py | Feeds the user messages from the specified jonsl file into an uncontrolled LLM |
| code/M2_generate_controlled.py | Feeds the user messages from the specified jonsl file into an LLM, once with the infavor.txt system prompt and once with the against.txt system prompt located in the working directory that is specified. |
| code/M3_filter_responses.py | Take a file with LLM responses and filteres paragraphs that do not match the target stance. Uses the specified prompt for stance detection. |
| ./run_experiment_<model_name>.sh | Script used to perform all experiments for the respective model. See these script for examples on the arguments for all above mentioned scripts |
| ./topic_descriptions.json | Contains the natural language description of the topics used to fill the prompt templates. Specify this as an argument in the above scripts | 

### Setting up Ollama
In order to run the scripts an working ollama installation must be present on your system.
See [here](https://github.com/ollama/ollama) for installation process.
Befor running the scripts, pull the respective LLMs with ollama, eg. "ollama pull gemma3:4b".

### Reproduction of experiments
First create an python3.11 virtual environment and install the requirements.txt.
Then place the input data in the data folder (call it data_cleaned.jsonl or adjust the data variable in the sh script).
Then run the sh scripts.
If the GPT script is used, fist export the system variabel OPENAI_API_KEY filled with your openAI subscription key.
In case you are unsure about the arguments of the above mentioned scripts use "python3 <script_name> --help".