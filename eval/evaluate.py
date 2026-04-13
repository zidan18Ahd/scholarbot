from rouge_score import rouge_scorer
from bert_score import score as bert_score

def evaluate(predictions, references):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
    )
    r = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    for p, ref in zip(predictions, references):
        s = scorer.score(ref, p)
        for k in r:
            r[k].append(s[k].fmeasure)
    n = len(predictions)
    _, _, F1 = bert_score(predictions, references, lang='en')
    return {
        'rouge1':       round(sum(r['rouge1'])/n, 4),
        'rouge2':       round(sum(r['rouge2'])/n, 4),
        'rougeL':       round(sum(r['rougeL'])/n, 4),
        'bertscore_f1': round(F1.mean().item(), 4),
    }
