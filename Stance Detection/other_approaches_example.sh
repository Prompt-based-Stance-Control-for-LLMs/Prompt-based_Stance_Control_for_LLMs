## Example for training and predicting with the GermanBert baseline
python3 ./code/M1_fine_tune_lm.py google-bert/bert-base-german-cased ./data/train.jsonl ./data/valid.jsonl ./exp_GermanBert/ --epochs 3 --lr 1e-5 --batch-size 16
python3 ./code/M1_predict_lm.py google-bert/bert-base-german-cased ./exp_GermanBert/final_model ./data/test.jsonl ./exp_GermanBert/test_predictions.jsonl

## Example for LLM-prepared-BERT
python3 ./code/M3_prepare_for_bert_with_llm.py mistral-small ./data/train_5ksample.jsonl ./exp_LLM_preped+BERT/prepared_train.jsonl
# here the same for the valid.jsonl and the test.jsonl
# And then train a GermanBert baseline on it as shown above


## Example for Bert-then-LLM
python3 ./code/M4_combine_bert_and_llm.py ./exp_LLM-based_GermanExample/test.qwen.jsonl ./exp_GermanBert/test_predictions.jsonl ./exp_BERT_then_LLM/test_predictions.jsonl --threshold 0.9

