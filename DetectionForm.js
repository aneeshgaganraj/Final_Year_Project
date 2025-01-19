import React, { useState } from 'react';
import { TextField, Button, Typography, Box } from '@mui/material';

const DetectionForm = () => {
  const [claim, setClaim] = useState('');
  const [result, setResult] = useState(null);

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ claim }),
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  return (
    <Box
      component="form"
      onSubmit={handleSubmit}
      sx={{
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
        maxWidth: 400,
        margin: 'auto',
        mt: 4,
      }}
    >
      <Typography variant="h4" align="center" gutterBottom>
        Detection Form
      </Typography>
      <TextField
        label="Enter Claim"
        variant="outlined"
        value={claim}
        onChange={(e) => setClaim(e.target.value)}
        required
      />
      <Button type="submit" variant="contained" color="primary">
        Submit
      </Button>
      {result && (<>
                <Box mt={2}>
                <Typography variant="h6">Prediction:</Typography>
                <Typography>{result.prediction}</Typography>
              </Box>
        <Box mt={2}>
          <Typography variant="h6">Explanation:</Typography>
          <Typography>{result.explanation}</Typography>
        </Box>
        </>
      )}
    </Box>
  );
};

export default DetectionForm;