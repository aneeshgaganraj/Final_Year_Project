import os
import torch
from datasets import load_dataset, DatasetDict
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

def load_model():
    """
    Load the saved fine-tuned BERT model and tokenizer for inference.
    """
    saved_model_path = './saved_model'  # Adjust this path if the model is saved elsewhere

    # Verify the saved model directory exists
    if not os.path.exists(saved_model_path):
        raise FileNotFoundError(f"The saved model directory does not exist at: {saved_model_path}")

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(saved_model_path)
    model = BertForSequenceClassification.from_pretrained(saved_model_path)

    print("Model and tokenizer loaded successfully.")
    return {"model": model, "tokenizer": tokenizer}

def train_model():
    """
    Fine-tune the BERT model on the Fin-Fact dataset for financial misinformation detection
    and save the trained model.
    """
    # Load the dataset
    dataset = load_dataset("amanrangapur/Fin-Fact")

    # Map string labels to integers
    def map_labels(example):
        label_mapping = {"true": 0, "false": 1, "not enough information": 2, "neutral": 3}
        example['label'] = label_mapping[example['label']]
        return example

    dataset = dataset.map(map_labels)

    # Split into train and test subsets (if test is not already available)
    train_test_split = dataset['train'].train_test_split(test_size=0.2, seed=42)
    dataset = DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['claim'],
            truncation=True,
            padding="max_length",  # Ensures all sequences are padded to the same length
            max_length=64          # Set a consistent max length (adjust as needed)
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Load the pre-trained BERT model with updated num_labels
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)  # Updated to 4 labels

    # Define training arguments with updated eval_strategy
    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy="epoch",  # Updated from evaluation_strategy
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_steps=500,  # Save the model periodically
        save_total_limit=2  # Keep only the last 2 checkpoints
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test']
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    save_path = './saved_model'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)  # Ensure save path exists

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"Model and tokenizer have been saved to {save_path}.")

def predict_claim(model_data, claim):
    """
    Predict the label for a given financial claim using the fine-tuned model.
    
    Args:
        model_data (dict): A dictionary containing the model and tokenizer.
        claim (str): The financial claim to predict.
    
    Returns:
        tuple: A tuple containing the predicted label and an explanation.
    """
    tokenizer = model_data['tokenizer']
    model = model_data['model']

    # Tokenize the claim
    inputs = tokenizer(claim, return_tensors="pt", truncation=True, padding="max_length", max_length=64)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract prediction
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()

    # Define class labels (updated for 4 labels)
    classes = ['True', 'False', 'Not Enough Information', 'Neutral']
    
    prediction = classes[predicted_class_id]

    # Generate explanation
    explanation = f"The claim was classified as '{prediction}' based on the input text."

    return prediction, explanation