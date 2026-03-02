from collections import Counter


def dict2probs(d):
    tot = sum(d.values())
    return {k:100*v/tot for k,v in d.items()}

def sum_dicts(ds):
    o = dict()
    for d in ds:
        for k,v in d.items():
            try:
                o[k] += v
            except KeyError:
                o[k] = v
    return o

def sub_dicts(a,b):
    o = dict()
    for k in a.keys():
        o[k] = a[k]-b[k]
    return o

def eval_whole(ds):
    return dict2probs(Counter(ds))

def eval_paragraph(ds):
    return dict2probs(sum_dicts(ds.apply(lambda x:dict2probs(Counter([l for p,l in x])))))
