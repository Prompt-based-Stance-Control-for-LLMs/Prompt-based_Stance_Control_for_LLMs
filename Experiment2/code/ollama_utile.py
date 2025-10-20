import time


def classify(text, topic, model, num_retries, allowed_classes, total_oai_tokens=None):
    result,format_flag,retry_i=None,None,None
    for retry_i in range(num_retries):
        # predict current question-comment-pair
        res = model.invoke({"question":topic , "text":text})
        if total_oai_tokens is not None:
            total_oai_tokens["input_tokens"] += res.usage_metadata["input_tokens"]
            total_oai_tokens["output_tokens"] += res.usage_metadata["output_tokens"]
            res = res.content
#            time.sleep(3)
        res = res.strip()
        format_flag = (res not in allowed_classes)
        #
        print(format_flag, res)
        if (format_flag is False):
            result = res
            break
        
    return result, format_flag, retry_i