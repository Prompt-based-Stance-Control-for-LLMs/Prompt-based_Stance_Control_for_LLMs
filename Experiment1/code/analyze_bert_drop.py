from data import load_xstance, pd_read_jsonl




train = load_xstance("../data/xstance/train.jsonl")
valid = load_xstance("../data/xstance/valid.jsonl")
test = load_xstance("../data/xstance/test.jsonl")

##
train_qid = set(train["question_id"])
valid_qid = set(valid["question_id"])
test_qid = set(test["question_id"])

train_ids_in_valid = valid_qid.intersection(train_qid)
train_ids_in_test = test_qid.intersection(train_qid)
valid_ids_missing_from_train = valid_qid-train_qid
test_ids_missing_from_train = test_qid-train_qid
test_ids_in_valid =  test_qid.intersection(valid_qid)
print("Valid missing from train:", len(valid_ids_missing_from_train))
print("Test missing from train:", len(test_ids_missing_from_train))
print()
print("Test in valid:", len(test_ids_in_valid))
print()

##
valid_predictions = pd_read_jsonl("exp1_german_bert/6348.valid.predictions.jsonl").set_index("id")
valid_predictions = valid_predictions.join(valid.set_index("id")).reset_index()
test_predictions = pd_read_jsonl("exp1_german_bert/6348.test.predictions.jsonl").set_index("id")
test_predictions = test_predictions.join(test.set_index("id")).reset_index()


##
valid_unseen_train = valid_predictions[valid_predictions["question_id"].apply(lambda x:x in train_ids_in_valid)]
#
test_seen_valid = test_predictions[test_predictions["question_id"].apply(lambda x:x in test_ids_in_valid)]
test_unseen_train = test_predictions[test_predictions["question_id"].apply(lambda x:x in test_ids_missing_from_train)]
test_seen_train = test_predictions[test_predictions["question_id"].apply(lambda x:x not in test_ids_missing_from_train)]

print("== Valid Un-Seen in training: ==========================")
print(valid_unseen_train)
print("== Test Seen in valid: ==========================")
print(test_seen_valid)
print("== Test Un-Seen in training: =========================")
print(test_unseen_train)
print(test_unseen_train["test_set"].unique())
print("== Test Seen in training: =========================")
print(test_seen_train)
print(test_seen_train["test_set"].unique())


##
wrong_test = test_predictions[test_predictions["pred_label"]!=test_predictions["label"]]
correct_test = test_predictions[test_predictions["pred_label"]==test_predictions["label"]]


print("\n\nWrong predicitons by type of example (Test set):")
print(wrong_test[["id", "question_id", "label", "pred_label", "test_set"]].groupby("test_set").size())

print("\n\nCorrect predicitons by type of example (Test set):")
print(correct_test[["id", "question_id", "label", "pred_label", "test_set"]].groupby("test_set").size())
