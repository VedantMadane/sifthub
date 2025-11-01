import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { AlertCircle, CheckCircle, Download, PlayCircle, Loader2 } from 'lucide-react';

const EmbeddingEvaluationSystem = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [evaluationResults, setEvaluationResults] = useState(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [groundTruth, setGroundTruth] = useState({});
  const [evaluationStatus, setEvaluationStatus] = useState('idle'); // idle, running, completed, error

  // API base URL - adjust this to match your backend
  const API_BASE_URL = 'http://127.0.0.1:8001';

  // Fetch ground truth data on component mount
  useEffect(() => {
    fetchGroundTruth();
  }, []);

  const fetchGroundTruth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/ground-truth`);
      if (response.ok) {
        const data = await response.json();
        // Convert API response to the format expected by the frontend
        const convertedData = {};
        Object.keys(data).forEach(key => {
          convertedData[key] = {
            question: data[key].question,
            relevantChunks: data[key].relevantChunks.map(chunk => ({
              id: chunk.id,
              text: chunk.text,
              section: chunk.section,
              relevance: 1.0 // Default relevance, could be enhanced if provided by API
            }))
          };
        });
        setGroundTruth(convertedData);
      }
    } catch (error) {
      console.error('Error fetching ground truth:', error);
    }
  };

  const startEvaluation = async () => {
    setIsEvaluating(true);
    setEvaluationStatus('running');
    
    try {
      // Start the evaluation
      const response = await fetch(`${API_BASE_URL}/evaluate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error('Failed to start evaluation');
      }
      
      // Poll for results
      pollForResults();
    } catch (error) {
      console.error('Error starting evaluation:', error);
      setIsEvaluating(false);
      setEvaluationStatus('error');
    }
  };

  const pollForResults = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/results`);
      const data = await response.json();
      
      if (data.status === 'completed') {
        setEvaluationResults(data.results);
        setIsEvaluating(false);
        setEvaluationStatus('completed');
      } else if (data.status === 'error') {
        setIsEvaluating(false);
        setEvaluationStatus('error');
      } else {
        // Still running, poll again in 2 seconds
        setTimeout(pollForResults, 2000);
      }
    } catch (error) {
      console.error('Error polling for results:', error);
      setIsEvaluating(false);
      setEvaluationStatus('error');
    }
  };

  // Simulated embedding models with different characteristics
  const embeddingModels = [
    {
      name: "sentence-transformers/all-mpnet-base-v2",
      dimensions: 768,
      description: "General-purpose model, balanced performance",
      strengths: "Good semantic understanding, balanced speed/accuracy"
    },
    {
      name: "BAAI/bge-large-en-v1.5",
      dimensions: 1024,
      description: "High-performing retrieval model",
      strengths: "Excellent for retrieval tasks, strong on technical content"
    },
    {
      name: "intfloat/e5-large-v2",
      dimensions: 1024,
      description: "Instruction-following embedding model",
      strengths: "Handles queries and passages differently, great for Q&A"
    }
  ];

  const MetricBadge = ({ value, label }) => (
    <div className="flex flex-col items-center p-2 bg-gray-50 rounded">
      <span className="text-2xl font-bold text-blue-600">{value.toFixed(3)}</span>
      <span className="text-xs text-gray-600">{label}</span>
    </div>
  );

  return (
    <div className="w-full max-w-7xl mx-auto p-4">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Embedding Model Evaluation System</h1>
        <p className="text-gray-600">Comprehensive evaluation framework for selecting the best embedding model for RAG tasks</p>
      </div>

      {/* Navigation Tabs */}
      <div className="flex gap-2 mb-6 border-b">
        {['overview', 'ground-truth', 'models', 'evaluation', 'results'].map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 font-medium transition-colors ${
              activeTab === tab
                ? 'border-b-2 border-blue-600 text-blue-600'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            {tab.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Evaluation Strategy</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h3 className="font-semibold mb-2">1. Ground Truth Creation</h3>
                <p className="text-sm text-gray-600">
                  Manually curated relevant chunks for each question from the source document, 
                  with relevance scores (0-1) based on how directly they answer the question.
                </p>
              </div>
              <div>
                <h3 className="font-semibold mb-2">2. Evaluation Metrics</h3>
                <ul className="text-sm text-gray-600 space-y-1 list-disc list-inside">
                  <li><strong>Precision@K:</strong> Ratio of relevant chunks in top K results</li>
                  <li><strong>Recall@K:</strong> Ratio of relevant chunks retrieved out of all relevant</li>
                  <li><strong>F1 Score:</strong> Harmonic mean of precision and recall</li>
                  <li><strong>MRR (Mean Reciprocal Rank):</strong> Position of first relevant result</li>
                  <li><strong>NDCG (Normalized Discounted Cumulative Gain):</strong> Ranking quality with relevance weighting</li>
                </ul>
              </div>
              <div>
                <h3 className="font-semibold mb-2">3. Model Selection Criteria</h3>
                <ul className="text-sm text-gray-600 space-y-1 list-disc list-inside">
                  <li>Performance on technical enterprise documentation</li>
                  <li>Semantic understanding of security and compliance terminology</li>
                  <li>Balance between accuracy and computational efficiency</li>
                  <li>Strong community support and documentation</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Ground Truth Tab */}
      {activeTab === 'ground-truth' && (
        <div className="space-y-4">
          {Object.entries(groundTruth).map(([qKey, qData]) => (
            <Card key={qKey}>
              <CardHeader>
                <CardTitle className="text-lg">Question {qKey.substring(1)}</CardTitle>
                <p className="text-sm text-gray-600">{qData.question}</p>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {qData.relevantChunks.map((chunk, idx) => (
                    <div key={chunk.id} className="p-3 bg-green-50 border border-green-200 rounded">
                      <div className="flex justify-between items-start mb-2">
                        <span className="text-xs font-semibold text-green-700">{chunk.section}</span>
                        <span className="text-xs bg-green-600 text-white px-2 py-1 rounded">
                          Relevance: {(chunk.relevance * 100).toFixed(0)}%
                        </span>
                      </div>
                      <p className="text-sm">{chunk.text}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Models Tab */}
      {activeTab === 'models' && (
        <div className="grid md:grid-cols-3 gap-4">
          {embeddingModels.map((model, idx) => (
            <Card key={idx}>
              <CardHeader>
                <CardTitle className="text-lg">{model.name}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div>
                  <span className="text-sm font-semibold">Dimensions:</span>
                  <span className="text-sm ml-2">{model.dimensions}</span>
                </div>
                <div>
                  <span className="text-sm font-semibold">Description:</span>
                  <p className="text-sm text-gray-600">{model.description}</p>
                </div>
                <div>
                  <span className="text-sm font-semibold">Strengths:</span>
                  <p className="text-sm text-gray-600">{model.strengths}</p>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Evaluation Tab */}
      {activeTab === 'evaluation' && (
        <Card>
          <CardHeader>
            <CardTitle>Run Evaluation</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-gray-600">
              This will evaluate all three embedding models against the ground truth dataset
              using the defined metrics. The evaluation will run on the backend and results will be displayed here.
            </p>
            <button
              onClick={startEvaluation}
              disabled={isEvaluating}
              className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 transition-colors"
            >
              {isEvaluating ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Evaluating...
                </>
              ) : (
                <>
                  <PlayCircle className="w-5 h-5" />
                  Run Evaluation
                </>
              )}
            </button>
            {evaluationStatus === 'error' && (
              <div className="text-red-600">
                Error occurred during evaluation. Please check the backend logs.
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Results Tab */}
      {activeTab === 'results' && evaluationResults && (
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Model Comparison Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left p-2">Model</th>
                      <th className="text-center p-2">Precision</th>
                      <th className="text-center p-2">Recall</th>
                      <th className="text-center p-2">F1</th>
                      <th className="text-center p-2">MRR</th>
                      <th className="text-center p-2">NDCG</th>
                    </tr>
                  </thead>
                  <tbody>
                    {evaluationResults.map((modelData, idx) => (
                      <tr key={idx} className="border-b hover:bg-gray-50">
                        <td className="p-2 font-medium">{modelData.model_name.split('/')[1]}</td>
                        <td className="text-center p-2">{modelData.average_metrics.precision_at_k.toFixed(3)}</td>
                        <td className="text-center p-2">{modelData.average_metrics.recall_at_k.toFixed(3)}</td>
                        <td className="text-center p-2">{modelData.average_metrics.f1_score.toFixed(3)}</td>
                        <td className="text-center p-2">{modelData.average_metrics.mrr.toFixed(3)}</td>
                        <td className="text-center p-2">{modelData.average_metrics.ndcg.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          {/* Detailed Results per Model */}
          {evaluationResults.map((modelData, idx) => (
            <Card key={idx}>
              <CardHeader>
                <CardTitle>{modelData.model_name}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-5 gap-2 mb-4">
                  <MetricBadge value={modelData.average_metrics.precision_at_k} label="Avg Precision" />
                  <MetricBadge value={modelData.average_metrics.recall_at_k} label="Avg Recall" />
                  <MetricBadge value={modelData.average_metrics.f1_score} label="Avg F1" />
                  <MetricBadge value={modelData.average_metrics.mrr} label="Avg MRR" />
                  <MetricBadge value={modelData.average_metrics.ndcg} label="Avg NDCG" />
                </div>
                <div className="space-y-2">
                  {Object.entries(modelData.per_question).map(([qKey, qData]) => (
                    <div key={qKey} className="p-3 bg-gray-50 rounded">
                      <div className="flex justify-between items-center mb-2">
                        <span className="font-medium">Question {qKey.substring(1)}</span>
                        <span className="text-sm text-gray-600">
                          F1: {qData.metrics.f1_score.toFixed(3)} | MRR: {qData.metrics.mrr.toFixed(3)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}

          {/* Conclusion */}
          <Card className="border-2 border-green-500">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <CheckCircle className="w-6 h-6 text-green-600" />
                Recommendation
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <p className="text-gray-700">
                Based on the comprehensive evaluation across all metrics, the recommended embedding model will be displayed here after evaluation.
              </p>
            </CardContent>
          </Card>
        </div>
      )}

      {!evaluationResults && activeTab === 'results' && (
        <Card>
          <CardContent className="py-12 text-center">
            <AlertCircle className="w-12 h-12 mx-auto mb-4 text-gray-400" />
            <p className="text-gray-600">No evaluation results yet. Run the evaluation first.</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default EmbeddingEvaluationSystem;