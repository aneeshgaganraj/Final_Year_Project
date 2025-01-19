from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from BARTScore.bart_score import BARTScorer  # Import BARTScorer class or function
import torch

def evaluate_metrics(predictions, references):
    """
    Evaluate the model's predictions using various metrics:
      - Accuracy
      - Precision
      - Recall
      - Micro-F1 Score
      - ROUGE (for explanation quality)
      - BERTScore (for explanation quality)
      - BARTScore (for explanation quality)

    Args:
        predictions (dict): A dictionary containing predicted labels and explanations.
                            Example: {'labels': [...], 'explanations': [...]}
        references (dict): A dictionary containing true labels and reference explanations.
                           Example: {'labels': [...], 'explanations': [...]}

    Returns:
        dict: A dictionary containing all evaluation metrics.
    """

    # Classification Metrics (Accuracy, Precision, Recall, Micro-F1)
    accuracy = accuracy_score(references['labels'], predictions['labels'])
    
    precision = precision_score(references['labels'], predictions['labels'], average='micro')
    
    recall = recall_score(references['labels'], predictions['labels'], average='micro')
    
    micro_f1 = f1_score(references['labels'], predictions['labels'], average='micro')

    # ROUGE Scores for Explanations
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge_scores = {
        "rouge1": [],
        "rouge2": [],
        "rougeL": []
    }

    for pred_explanation, ref_explanation in zip(predictions['explanations'], references['explanations']):
        scores = rouge_scorer_obj.score(pred_explanation, ref_explanation)
        rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
        rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
        rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)

    avg_rouge_scores = {
        "rouge1": sum(rouge_scores["rouge1"]) / len(rouge_scores["rouge1"]),
        "rouge2": sum(rouge_scores["rouge2"]) / len(rouge_scores["rouge2"]),
        "rougeL": sum(rouge_scores["rougeL"]) / len(rouge_scores["rougeL"])
    }

    # BERTScore for Explanations
    P, R, F1_bert = bert_score(predictions['explanations'], references['explanations'], lang="en")
    
    avg_bert_scores = {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1_bert.mean().item()
    }

    # BARTScore for Explanations (requires GPU or CPU with BART model loaded)
    bart_scorer = BARTScorer(device='cuda' if torch.cuda.is_available() else 'cpu', checkpoint='facebook/bart-large-cnn')
    
    bart_scores = bart_scorer.score(predictions['explanations'], references['explanations'])
    
    avg_bart_score = sum(bart_scores) / len(bart_scores)

    # Return all metrics in a dictionary
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "micro_f1": micro_f1,
        "avg_rouge_scores": avg_rouge_scores,
        "avg_bert_scores": avg_bert_scores,
        "avg_bart_score": avg_bart_score
    }