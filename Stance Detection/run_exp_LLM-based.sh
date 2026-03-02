code_location='./code/M2_predict_llm_ollama.py'
data_location='./data/valid.jsonl'
sauerkraut_model='hf.co/QuantFactory/SauerkrautLM-Nemo-12b-Instruct-GGUF:Q8_0'
laemmlein_model='hf.co/LSX-UniWue/LLaMmlein_1B_chat_selected'
mistral_model='mistral-small'
gemma_model='gemma2:9b'
qwen_model='qwen2.5:14b'


base_path='./exp_LLM-based_GermanSimple'
python3 $code_location $sauerkraut_model $data_location "$base_path/val.sauerkraut.jsonl" Zustimmung,Ablehnung "$base_path/system.txt" "$base_path/user.txt" --task-as-system --num-retries 5
python3 $code_location $mistral_model    $data_location "$base_path/val.mistral.jsonl"    Zustimmung,Ablehnung "$base_path/system.txt" "$base_path/user.txt" --task-as-system --num-retries 5
python3 $code_location $gemma_model      $data_location "$base_path/val.gemma.jsonl"      Zustimmung,Ablehnung "$base_path/system.txt" "$base_path/user.txt" --task-as-system --num-retries 5
python3 $code_location $qwen_model       $data_location "$base_path/val.qwen.jsonl"       Zustimmung,Ablehnung "$base_path/system.txt" "$base_path/user.txt" --task-as-system --num-retries 5
# not run to its end, since not working at all
#python3 $code_location $laemmlein_model  $data_location "$base_path/val.llammlein.jsonl"  Zustimmung,Ablehnung "$base_path/system.txt" "$base_path/user.txt" --task-as-system --num-retries 5


exa_path='./exp_LLM-based_GermanExample'
python3 $code_location $sauerkraut_model $data_location "$exa_path/val.sauerkraut.jsonl" Zustimmung,Ablehnung "$exa_path/system.txt" "$exa_path/user.txt" --task-as-system --num-retries 5
python3 $code_location $mistral_model    $data_location "$exa_path/val.mistral.jsonl"    Zustimmung,Ablehnung "$exa_path/system.txt" "$exa_path/user.txt" --task-as-system --num-retries 5
python3 $code_location $gemma_model      $data_location "$exa_path/val.gemma.jsonl"      Zustimmung,Ablehnung "$exa_path/system.txt" "$exa_path/user.txt" --task-as-system --num-retries 5
python3 $code_location $qwen_model       $data_location "$exa_path/val.qwen.jsonl"       Zustimmung,Ablehnung "$exa_path/system.txt" "$exa_path/user.txt" --task-as-system --num-retries 5
# not tested at all, since doenst even worked for german prompt
#python3 $code_location $laemmlein_model  $data_location "$exa_path/val.llammlein.jsonl"  Zustimmung,Ablehnung "$exa_path/system.txt" "$exa_path/user.txt" --task-as-system --num-retries 5


base_path='./exp_LLM-based_EnglishSimple'
python3 $code_location $sauerkraut_model $data_location "$base_path/val.sauerkraut.jsonl" favor,against "$base_path/system.txt" "$base_path/user.txt" --task-as-system --num-retries 5
python3 $code_location $mistral_model    $data_location "$base_path/val.mistral.jsonl"    favor,against "$base_path/system.txt" "$base_path/user.txt" --task-as-system --num-retries 5
python3 $code_location $gemma_model      $data_location "$base_path/val.gemma.jsonl"      favor,against "$base_path/system.txt" "$base_path/user.txt" --task-as-system --num-retries 5
python3 $code_location $qwen_model       $data_location "$base_path/val.qwen.jsonl"       favor,against "$base_path/system.txt" "$base_path/user.txt" --task-as-system --num-retries 5


exa_path='./exp_LLM-based_EnglishExample'
python3 $code_location $sauerkraut_model $data_location "$exa_path/val.sauerkraut.jsonl" favor,against "$exa_path/system.txt" "$exa_path/user.txt" --task-as-system --num-retries 5
python3 $code_location $mistral_model    $data_location "$exa_path/val.mistral.jsonl"    favor,against "$exa_path/system.txt" "$exa_path/user.txt" --task-as-system --num-retries 5
python3 $code_location $gemma_model      $data_location "$exa_path/val.gemma.jsonl"      favor,against "$exa_path/system.txt" "$exa_path/user.txt" --task-as-system --num-retries 5
python3 $code_location $qwen_model       $data_location "$exa_path/val.qwen.jsonl"       favor,against "$exa_path/system.txt" "$exa_path/user.txt" --task-as-system --num-retries 5

