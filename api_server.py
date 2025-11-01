"""
FastAPI server for the Embedding Model Evaluation System
"""
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import threading
import time
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our evaluation functions
from embedding_evaluation import run_evaluation, create_ground_truth, DocumentChunker, Chunk, GroundTruthDataset

app = FastAPI(title="Embedding Model Evaluation API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store evaluation results
evaluation_results = None
evaluation_status = "idle"  # idle, running, completed, error
evaluation_error = None

class EvaluationStatus(BaseModel):
    status: str
    error: Optional[str] = None

class ChunkInfo(BaseModel):
    id: str
    text: str
    section: str

class GroundTruthQuestion(BaseModel):
    question: str
    relevantChunks: List[ChunkInfo]

class EvaluationRequest(BaseModel):
    pass

@app.get("/")
async def root():
    return {"message": "Embedding Model Evaluation API"}

@app.get("/status")
async def get_status() -> EvaluationStatus:
    """Get the current evaluation status"""
    return EvaluationStatus(status=evaluation_status, error=evaluation_error)

@app.get("/ground-truth")
async def get_ground_truth() -> Dict[str, GroundTruthQuestion]:
    """Get the ground truth dataset"""
    ground_truth = create_ground_truth()
    result = {}
    
    # Define chunk data for reference
    chunker = DocumentChunker("ACME Enterprise Platform.docx")
    chunks = chunker.load_and_chunk()
    chunk_dict = {chunk.id: chunk for chunk in chunks}
    
    for q_id, q_data in ground_truth.ground_truth.items():
        relevant_chunks = []
        for chunk_id in q_data['relevant_chunks']:
            if chunk_id in chunk_dict:
                chunk = chunk_dict[chunk_id]
                relevant_chunks.append(ChunkInfo(
                    id=chunk.id,
                    text=chunk.text,
                    section=chunk.section
                ))
        
        result[q_id] = GroundTruthQuestion(
            question=q_data['question'],
            relevantChunks=relevant_chunks
        )
    
    return result

@app.post("/evaluate")
async def start_evaluation() -> EvaluationStatus:
    """Start the evaluation process"""
    global evaluation_status, evaluation_results, evaluation_error
    
    if evaluation_status == "running":
        return EvaluationStatus(status="running", error="Evaluation already in progress")
    
    evaluation_status = "running"
    evaluation_error = None
    
    def run_eval():
        global evaluation_status, evaluation_results, evaluation_error
        try:
            evaluation_results = run_evaluation()
            evaluation_status = "completed"
        except Exception as e:
            evaluation_error = str(e)
            evaluation_status = "error"
    
    # Run evaluation in a separate thread to avoid blocking
    thread = threading.Thread(target=run_eval)
    thread.start()
    
    return EvaluationStatus(status="running")

@app.get("/results")
async def get_results():
    """Get the evaluation results"""
    global evaluation_results, evaluation_status
    if evaluation_status != "completed":
        return {"status": evaluation_status, "results": None}
    return {"status": "completed", "results": evaluation_results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)