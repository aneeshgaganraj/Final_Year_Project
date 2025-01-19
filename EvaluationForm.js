import React, { useState } from 'react';
import { TextField, Button, Typography, Box } from '@mui/material';

const EvaluationForm = () => {
  const [predictionsLabels, setPredictionsLabels] = useState('');
  const [predictionsExplanations, setPredictionsExplanations] = useState('');
  const [referencesLabels, setReferencesLabels] = useState('');
  const [referencesExplanations, setReferencesExplanations] = useState('');
  const [evaluationResults, setEvaluationResults] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      // Convert plain text inputs into JSON format
      const predictions = {
        labels: predictionsLabels.split(',').map((label) => label.trim()),
        explanations: predictionsExplanations.split('\n').map((exp) => exp.trim()),
      };
      const references = {
        labels: referencesLabels.split(',').map((label) => label.trim()),
        explanations: referencesExplanations.split('\n').map((exp) => exp.trim()),
      };

      const response = await fetch('http://localhost:5000/evaluate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ predictions, references }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setEvaluationResults(data);
    } catch (error) {
      console.error('Error fetching evaluation results:', error);
      setErrorMessage('Failed to fetch evaluation results. Please check your input.');
    }
  };

  return (
    <Box component="form" onSubmit={handleSubmit} sx={{ maxWidth: 600, mx: 'auto', mt: 4 }}>
      <Typography variant="h4" gutterBottom>
        Evaluation Form
      </Typography>
      
      <TextField
        label="Predicted Labels (comma-separated)"
        fullWidth
        margin="normal"
        value={predictionsLabels}
        onChange={(e) => setPredictionsLabels(e.target.value)}
        required
      />
      
      <TextField
        label="Predicted Explanations (one per line)"
        fullWidth
        margin="normal"
        multiline
        rows={4}
        value={predictionsExplanations}
        onChange={(e) => setPredictionsExplanations(e.target.value)}
        required
      />
      
      <TextField
        label="Reference Labels (comma-separated)"
        fullWidth
        margin="normal"
        value={referencesLabels}
        onChange={(e) => setReferencesLabels(e.target.value)}
        required
      />
      
      <TextField
        label="Reference Explanations (one per line)"
        fullWidth
        margin="normal"
        multiline
        rows={4}
        value={referencesExplanations}
        onChange={(e) => setReferencesExplanations(e.target.value)}
        required
      />
      
      <Button type="submit" variant="contained" color="primary" sx={{ mt: 2 }}>
        Submit
      </Button>

      {errorMessage && (
        <Typography color="error" sx={{ mt: 2 }}>
          {errorMessage}
        </Typography>
      )}

      {evaluationResults && (
        <Box sx={{ mt: 4 }}>
          <Typography variant="h6">Evaluation Results:</Typography>
          <Typography>Accuracy: {evaluationResults.accuracy}</Typography>
          <Typography>Precision: {evaluationResults.precision}</Typography>
          <Typography>Recall: {evaluationResults.recall}</Typography>
          <Typography>Micro-F1: {evaluationResults.micro_f1}</Typography>
          <Typography>ROUGE-1: {evaluationResults.avg_rouge_scores.rouge1}</Typography>
          <Typography>ROUGE-2: {evaluationResults.avg_rouge_scores.rouge2}</Typography>
          <Typography>ROUGE-L: {evaluationResults.avg_rouge_scores.rougeL}</Typography>
          <Typography>BERT Precision: {evaluationResults.avg_bert_scores.precision}</Typography>
          <Typography>BERT Recall: {evaluationResults.avg_bert_scores.recall}</Typography>
          <Typography>BERT F1: {evaluationResults.avg_bert_scores.f1}</Typography>
          <Typography>BART Score: {evaluationResults.avg_bart_score}</Typography>
        </Box>
      )}
    </Box>
  );
};

export default EvaluationForm;