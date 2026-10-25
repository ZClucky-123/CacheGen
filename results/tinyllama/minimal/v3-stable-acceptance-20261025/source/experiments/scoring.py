"""Frozen QA scoring shared by protocols v2 and v3."""
import re
import string
from collections import Counter


def normalize(text):
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def extract_answer(text):
    # Match the author's scorer_e: strip leading LF, then take the first LF line.
    # Do not strip arbitrary Unicode separators or rewrite the raw generation.
    return text.lstrip('\n').split('\n', 1)[0]


def score_qa(prediction, answers):
    if not answers or any(not isinstance(a, str) for a in answers):
        raise ValueError('QA requires nonempty, predeclared answer aliases')
    p = normalize(extract_answer(prediction))
    scores = []
    for answer in answers:
        a = normalize(answer)
        common = sum((Counter(p.split()) & Counter(a.split())).values())
        f1 = 2 * common / (len(p.split()) + len(a.split())) if common else float(p == a)
        scores.append((float(p == a), f1))
    return {'em': max(x[0] for x in scores), 'f1': max(x[1] for x in scores)}


def phrase_match(prediction, answer):
    # Historical synthetic task diagnostic, separate from standard EM/F1.
    def words(value):
        return ' '.join(re.findall(r'\w+', value.casefold()))
    return bool(words(answer)) and f' {words(answer)} ' in f' {words(extract_answer(prediction))} '
