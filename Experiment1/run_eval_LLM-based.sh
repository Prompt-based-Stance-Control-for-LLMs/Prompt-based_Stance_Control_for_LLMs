code_location='./code/evaluate_predictions.py'
gold_location='./data/valid.jsonl'


base_path='./exp_LLM-based_GermanSimple'
python3 $code_location Zustimmung,Ablehnung $gold_location "$base_path/val.sauerkraut.jsonl" "$base_path/val.sauerkraut.results.xlsx"
python3 $code_location Zustimmung,Ablehnung $gold_location "$base_path/val.mistral.jsonl"    "$base_path/val.mistral.results.xlsx"
python3 $code_location Zustimmung,Ablehnung $gold_location "$base_path/val.gemma.jsonl"      "$base_path/val.gemma.results.xlsx"
python3 $code_location Zustimmung,Ablehnung $gold_location "$base_path/val.qwen.jsonl"       "$base_path/val.qwen.results.xlsx"

base_path='./exp_LLM-based_GermanExample'
python3 $code_location Zustimmung,Ablehnung $gold_location "$base_path/val.sauerkraut.jsonl" "$base_path/val.sauerkraut.results.xlsx"
python3 $code_location Zustimmung,Ablehnung $gold_location "$base_path/val.mistral.jsonl"    "$base_path/val.mistral.results.xlsx"
python3 $code_location Zustimmung,Ablehnung $gold_location "$base_path/val.gemma.jsonl"      "$base_path/val.gemma.results.xlsx"
python3 $code_location Zustimmung,Ablehnung $gold_location "$base_path/val.qwen.jsonl"       "$base_path/val.qwen.results.xlsx"


base_path='./exp_LLM-based_EnglishSimple'
python3 $code_location favor,against $gold_location "$base_path/val.sauerkraut.jsonl" "$base_path/val.sauerkraut.results.xlsx"
python3 $code_location favor,against $gold_location "$base_path/val.mistral.jsonl"    "$base_path/val.mistral.results.xlsx"
python3 $code_location favor,against $gold_location "$base_path/val.gemma.jsonl"      "$base_path/val.gemma.results.xlsx"
python3 $code_location favor,against $gold_location "$base_path/val.qwen.jsonl"       "$base_path/val.qwen.results.xlsx"

base_path='./exp_LLM-based_EnglishExample'
python3 $code_location favor,against $gold_location "$base_path/val.sauerkraut.jsonl" "$base_path/val.sauerkraut.results.xlsx"
python3 $code_location favor,against $gold_location "$base_path/val.mistral.jsonl"    "$base_path/val.mistral.results.xlsx"
python3 $code_location favor,against $gold_location "$base_path/val.gemma.jsonl"      "$base_path/val.gemma.results.xlsx"
python3 $code_location favor,against $gold_location "$base_path/val.qwen.jsonl"       "$base_path/val.qwen.results.xlsx"