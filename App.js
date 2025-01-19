import React from 'react';
import './App.css';
import DetectionForm from './components/DetectionForm';
import EvaluationForm from './components/EvaluationForm';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Financial Misinformation Detection</h1>
      </header>
      
      {/* Container for side-by-side layout */}
      <div className="form-container">
        <DetectionForm />
        <EvaluationForm />
      </div>
    </div>
  );
}

export default App;