# Paths
code_vanilla="./code/M1_generate_vanilla.py"
code_controlled="./code/M2_generate_controlled.py"
code_filter="./code/M3_filter_responses.py"
code_eval="./code/E_stance_detection.py"
topics="./topic_descriptions.json"
filter_prompt="./evaluation_labels/system.txt"

# Settings
data="data_cleaned"
#
gen_model="gpt-3.5-turbo"
filter_model="gpt-3.5-turbo"
model_name="gpt_3_5_turbo" ## needed since outputed files from below scripts will replace - and : with _ in the model name
#
eval_model="qwen2.5:14b"


## Produces a jonsl file in with llm generations in the working directory
## Output file will be named by using the generation models name and input data file name
## Prompt used to controll the llms is expected to be in a system.txt/system_infavor.txt/system_against.txt file in the working directory
#       code_location       llm_model   input_data           working_directory        topic_descrption
python3 $code_vanilla       $gen_model  "data/$data.jsonl"   ./responses_Vanilla                        --openai
python3 $code_controlled    $gen_model  "data/$data.jsonl"   ./responses_Basic        $topics           --openai
python3 $code_controlled    $gen_model  "data/$data.jsonl"   ./responses_Extended     $topics           --openai


## Produces a jonsl file in with filtered generations in the working directory
## Output file will be named like the input_data with an ".filtered." added to the name
#       code_location       llm_model     input_data                                       working_directory        topic_descrption     prompt_for_stance_detection
python3 $code_filter        $filter_model "responses_Basic/$data.$model_name.jsonl"        ./responses_Basic        $topics              $filter_prompt                --openai
python3 $code_filter        $filter_model "responses_Extended/$data.$model_name.jsonl"     ./responses_Extended     $topics              $filter_prompt                --openai


## Produces a jonsl file with stance detection results per example in the working_directory
## Prompt for evaluation (stance detection) is expected to be in a file system.txt in working_directory
## Output file will be named like the input_data with an ".evaluated." added to the name

#       code_location   llm_model   input_data                                              working_directory    topic_descrption
python3 $code_eval      $eval_model "responses_Vanilla/$data.$model_name.jsonl"             ./evaluation_labels  $topics  --num-retries 5
python3 $code_eval      $eval_model "responses_Basic/$data.$model_name.jsonl"               ./evaluation_labels  $topics  --num-retries 5
python3 $code_eval      $eval_model "responses_Extended/$data.$model_name.jsonl"            ./evaluation_labels  $topics  --num-retries 5
#
python3 $code_eval      $eval_model "responses_Basic/$data.$model_name.filtered.jsonl"      ./evaluation_labels  $topics --filtered "./evaluation_labels/responses_Basic.$data.$model_name.evaluated.jsonl"      --num-retries 5
python3 $code_eval      $eval_model "responses_Extended/$data.$model_name.filtered.jsonl"   ./evaluation_labels  $topics --filtered "./evaluation_labels/responses_Extended.$data.$model_name.evaluated.jsonl"   --num-retries 5
