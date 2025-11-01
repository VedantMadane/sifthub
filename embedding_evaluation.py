"""
Embedding Model Evaluation System for RAG Tasks
================================================
This script evaluates multiple embedding models for retrieval accuracy
on enterprise documentation using comprehensive metrics.

Requirements:
pip install sentence-transformers numpy scikit-learn pandas matplotlib seaborn
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import json
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import asyncio
import threading
import time

@dataclass
class Chunk:
    """Represents a text chunk from the document"""
    id: str
    text: str
    section: str
    relevance: float = 0.0  # Ground truth relevance (0-1)

@dataclass
class RetrievalResult:
    """Represents a retrieved chunk with similarity score"""
    chunk_id: str
    score: float
    rank: int

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    precision_at_k: float
    recall_at_k: float
    f1_score: float
    mrr: float
    ndcg: float
    map_score: float  # Mean Average Precision

class DocumentChunker:
    """Chunks the document into semantically meaningful segments"""
    
    def __init__(self, document_path: str):
        self.document_path = document_path
        self.chunks = []
    
    def load_and_chunk(self) -> List[Chunk]:
        """
        Load document and create chunks based on sections.
        In practice, you'd implement sophisticated chunking strategies.
        """
        # For this example, we'll define chunks manually based on the document
        # In production, you'd parse the document and chunk intelligently
        
        chunks = [
            # Mobile Security Controls
            Chunk(
                id="mobile_security_1",
                text="Mobile-Specific Security Controls: Through your UEM provider, you can enforce a wide range of granular security policies on the Acme mobile app including Block Copy/Paste to prevent users from copying data out of the Acme app and pasting it into unmanaged personal apps, Control File Downloads to block users from downloading files to the local device storage, Prevent Screen Capture to disable screenshots, and Remote Wipe to remotely delete all Acme data if a device is lost or stolen.",
                section="15.2 Mobile-Specific Security Controls"
            ),
            Chunk(
                id="mam_1",
                text="MAM (App Protection Policies): For BYOD scenarios, MAM allows administrators to apply security controls specifically to the Acme application and its data, without managing the user's entire personal device. This 'containerization' approach protects corporate data while respecting employee privacy.",
                section="15.1 MDM & MAM Integration"
            ),
            Chunk(
                id="mdm_general",
                text="The Acme mobile application for iOS and Android is fully compatible with leading Unified Endpoint Management (UEM) solutions, supporting both Mobile Device Management (MDM) and Mobile Application Management (MAM) policies.",
                section="15.1 MDM & MAM Integration"
            ),
            
            # Zero Trust Architecture
            Chunk(
                id="zero_trust_1",
                text="Zero Trust Architecture Philosophy: Acme's infrastructure and application architecture are guided by a Zero Trust philosophy. This means we do not automatically trust any user or service, whether internal or external to our network. Every access request is authenticated, authorized, and encrypted before being granted.",
                section="12.1 Zero Trust Architecture Philosophy"
            ),
            Chunk(
                id="zero_trust_2",
                text="Key principles of Zero Trust implementation include: Identity as the Perimeter where access controls are based on verified user and device identity not network location, Least Privilege Access where services and employees are granted the absolute minimum level of access required, Micro-segmentation where the production network is segmented into smaller isolated zones, and Continuous Verification where user and device state is continuously monitored.",
                section="12.1 Zero Trust Architecture Philosophy"
            ),
            Chunk(
                id="security_general",
                text="Security is not a feature at Acme; it is the foundation upon which our platform is built. We employ a defense-in-depth strategy, layering multiple physical, technical, and administrative controls to protect customer data.",
                section="5. Enterprise-Grade Security"
            ),
            
            # Audit Logs and SIEM
            Chunk(
                id="audit_logs_1",
                text="Comprehensive Audit Logs for SIEM Integration: Acme provides detailed, human-readable audit logs that capture a wide range of activities including User Activity, Administrative Actions, Channel & File Activity, and Access & Export events. All audit events are accessible via a dedicated, secure Audit Log API.",
                section="13.3 Comprehensive Audit Logs for SIEM Integration"
            ),
            Chunk(
                id="siem_integration_1",
                text="SIEM Integration: We provide pre-built connectors and support standardized formats (e.g., CEF, LEEF) for seamless integration with major Security Information and Event Management (SIEM) platforms like Splunk, QRadar, and Microsoft Sentinel. This allows your security team to correlate Acme activity with data from other systems in a single pane of glass.",
                section="13.3 Comprehensive Audit Logs for SIEM Integration"
            ),
            Chunk(
                id="compliance_general",
                text="Acme is committed to helping our customers meet their complex regulatory and data governance obligations through comprehensive portfolio of third-party certifications and attestations.",
                section="6. Compliance and Data Governance"
            ),
            
            # Information Barriers / Ethical Walls
            Chunk(
                id="ethical_walls_1",
                text="Ethical Walls & Information Barriers: For organizations in finance, legal, and other regulated industries, Acme provides an Information Barriers feature (also known as Ethical Walls). This allows compliance administrators to prevent specific groups of users from communicating with each other when a conflict of interest exists.",
                section="13.2 Ethical Walls & Information Barriers"
            ),
            Chunk(
                id="ethical_walls_2",
                text="When an Information Barrier policy is active between two groups: Users in one group cannot initiate a Direct Message with users in the other group, Users cannot be added to the same private channel, and Users from one group will not see users from the other group in search results or @mentions. This feature is critical for preventing insider trading and adhering to regulations like FINRA Rule 2241.",
                section="13.2 Ethical Walls & Information Barriers"
            ),
            Chunk(
                id="dlp_feature",
                text="Data Loss Prevention (DLP): Acme helps prevent the inadvertent or malicious exfiltration of sensitive data through DLP Integration with leading Cloud Access Security Broker (CASB) and DLP solutions.",
                section="6.4 Data Loss Prevention (DLP)"
            ),
            
            # AI Ethics and Governance
            Chunk(
                id="ai_ethics_1",
                text="Acme's AI Ethics Framework: The development of all Acme Intelligence features is governed by a strict internal AI Ethics Framework overseen by a cross-functional AI Ethics Council, based on four core principles: Fairness & Impartiality, Transparency & Explainability, Accountability & Human-in-the-Loop, and Privacy & Security by Design.",
                section="14.1 Acme's AI Ethics Framework"
            ),
            Chunk(
                id="ai_governance_1",
                text="AI Model Governance and Validation: We have a strict policy against using any customer's message or file content to train our global, multi-tenant AI models. Models are trained on anonymized, aggregated usage data or on publicly available and permissively licensed datasets. Before release, models undergo Bias Testing & Red Teaming to detect and mitigate performance disparities, and Continuous Monitoring of model performance in production.",
                section="14.2 AI Model Governance and Validation"
            ),
            Chunk(
                id="ai_general",
                text="Acme Intelligence is our suite of proprietary AI and machine learning features designed to augment human capability, reduce information overload, and surface critical insights from your organization's conversational data.",
                section="4. Acme Intelligence: The Future of Work"
            ),
        ]
        
        self.chunks = chunks
        return chunks

class GroundTruthDataset:
    """Manages ground truth data for evaluation"""
    
    def __init__(self):
        self.ground_truth = {}
    
    def add_question(self, question_id: str, question: str, relevant_chunk_ids: List[str], 
                    chunk_relevances: Dict[str, float]):
        """Add a question with its relevant chunks and relevance scores"""
        self.ground_truth[question_id] = {
            'question': question,
            'relevant_chunks': relevant_chunk_ids,
            'relevances': chunk_relevances
        }
    
    def get_relevant_chunks(self, question_id: str) -> List[str]:
        """Get list of relevant chunk IDs for a question"""
        return self.ground_truth[question_id]['relevant_chunks']
    
    def get_relevance_score(self, question_id: str, chunk_id: str) -> float:
        """Get relevance score for a specific chunk"""
        return self.ground_truth[question_id]['relevances'].get(chunk_id, 0.0)

class EmbeddingModelEvaluator:
    """Main evaluation class for embedding models"""
    
    def __init__(self, model_name: str, chunks: List[Chunk]):
        self.model_name = model_name
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        self.chunk_embeddings = None
        
        print(f"Loading model: {model_name}")
        self._encode_chunks()
    
    def _encode_chunks(self):
        """Encode all chunks using the embedding model"""
        chunk_texts = [chunk.text for chunk in self.chunks]
        print(f"Encoding {len(chunk_texts)} chunks...")
        self.chunk_embeddings = self.model.encode(chunk_texts, show_progress_bar=True)
        print(f"Encoding complete. Shape: {self.chunk_embeddings.shape}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve top-k most similar chunks for a query"""
        query_embedding = self.model.encode([query])[0]
        
        # Calculate cosine similarity
        similarities = cosine_similarity([query_embedding], self.chunk_embeddings)[0]
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Create retrieval results
        results = []
        for rank, idx in enumerate(top_k_indices, 1):
            results.append(RetrievalResult(
                chunk_id=self.chunks[idx].id,
                score=float(similarities[idx]),
                rank=rank
            ))
        
        return results
    
    def calculate_metrics(self, retrieved: List[RetrievalResult], 
                         ground_truth_ids: List[str],
                         relevance_scores: Dict[str, float]) -> EvaluationMetrics:
        """Calculate comprehensive evaluation metrics"""
        
        retrieved_ids = [r.chunk_id for r in retrieved]
        k = len(retrieved)
        
        # Precision@K and Recall@K
        relevant_retrieved = set(retrieved_ids) & set(ground_truth_ids)
        precision = len(relevant_retrieved) / k if k > 0 else 0.0
        recall = len(relevant_retrieved) / len(ground_truth_ids) if ground_truth_ids else 0.0
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        for result in retrieved:
            if result.chunk_id in ground_truth_ids:
                mrr = 1.0 / result.rank
                break
        
        # Normalized Discounted Cumulative Gain (NDCG)
        dcg = 0.0
        for result in retrieved:
            relevance = relevance_scores.get(result.chunk_id, 0.0)
            dcg += relevance / np.log2(result.rank + 1)
        
        # Ideal DCG
        sorted_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(rank + 2) for rank, rel in enumerate(sorted_relevances))
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        # Mean Average Precision (MAP)
        average_precision = 0.0
        relevant_count = 0
        for result in retrieved:
            if result.chunk_id in ground_truth_ids:
                relevant_count += 1
                precision_at_i = relevant_count / result.rank
                average_precision += precision_at_i
        
        map_score = average_precision / len(ground_truth_ids) if ground_truth_ids else 0.0
        
        return EvaluationMetrics(
            precision_at_k=precision,
            recall_at_k=recall,
            f1_score=f1,
            mrr=mrr,
            ndcg=ndcg,
            map_score=map_score
        )
    
    def evaluate_on_dataset(self, ground_truth: GroundTruthDataset, 
                           top_k: int = 3) -> Dict:
        """Evaluate model on entire ground truth dataset"""
        results = {}
        all_metrics = []
        
        for q_id, q_data in ground_truth.ground_truth.items():
            question = q_data['question']
            print(f"\nEvaluating Question {q_id}: {question[:80]}...")
            
            # Retrieve chunks
            retrieved = self.retrieve(question, top_k=top_k)
            
            # Calculate metrics
            metrics = self.calculate_metrics(
                retrieved,
                q_data['relevant_chunks'],
                q_data['relevances']
            )
            
            results[q_id] = {
                'question': question,
                'retrieved': retrieved,
                'metrics': metrics
            }
            all_metrics.append(metrics)
            
            # Print retrieved chunks
            print(f"  Retrieved chunks: {[r.chunk_id for r in retrieved]}")
            print(f"  Metrics: P={metrics.precision_at_k:.3f} R={metrics.recall_at_k:.3f} "
                  f"F1={metrics.f1_score:.3f} MRR={metrics.mrr:.3f} NDCG={metrics.ndcg:.3f}")
        
        # Calculate average metrics
        avg_metrics = EvaluationMetrics(
            precision_at_k=np.mean([m.precision_at_k for m in all_metrics]),
            recall_at_k=np.mean([m.recall_at_k for m in all_metrics]),
            f1_score=np.mean([m.f1_score for m in all_metrics]),
            mrr=np.mean([m.mrr for m in all_metrics]),
            ndcg=np.mean([m.ndcg for m in all_metrics]),
            map_score=np.mean([m.map_score for m in all_metrics])
        )
        
        return {
            'model_name': self.model_name,
            'per_question': results,
            'average_metrics': avg_metrics
        }

def create_ground_truth() -> GroundTruthDataset:
    """Create ground truth dataset with manually curated relevant chunks"""
    gt = GroundTruthDataset()
    
    # Question 1: Mobile device security controls
    gt.add_question(
        'q1',
        "Describe the specific policies and controls available to prevent data loss and control information flow from the Acme application on unmanaged, employee-owned mobile devices. Detail how data can be contained within the application and what actions can be taken if a device is compromised or lost.",
        ['mobile_security_1', 'mam_1'],
        {'mobile_security_1': 1.0, 'mam_1': 0.95}
    )
    
    # Question 2: Zero Trust architecture
    gt.add_question(
        'q2',
        "What is Acme's guiding architectural philosophy for security? Describe the core principles of this architecture, such as how it redefines the security perimeter and its approach to access control.",
        ['zero_trust_1', 'zero_trust_2'],
        {'zero_trust_1': 1.0, 'zero_trust_2': 0.98}
    )
    
    # Question 3: Audit logs and SIEM integration
    gt.add_question(
        'q3',
        "Our security operations team requires deep integration for monitoring user and administrative activity. Describe the platform's native capabilities for exporting detailed audit logs and the specific mechanisms or connectors provided for integration with enterprise Security Information and Event Management (SIEM) platforms.",
        ['audit_logs_1', 'siem_integration_1'],
        {'audit_logs_1': 1.0, 'siem_integration_1': 0.99}
    )
    
    # Question 4: Information Barriers / Ethical Walls
    gt.add_question(
        'q4',
        "For compliance purposes, we must enforce policies that actively prevent communication between specific user groups. What platform feature allows an administrator to create such a policy, and what is the user experience for individuals in groups where this communication control is enforced?",
        ['ethical_walls_1', 'ethical_walls_2'],
        {'ethical_walls_1': 1.0, 'ethical_walls_2': 0.97}
    )
    
    # Question 5: AI Ethics and Governance
    gt.add_question(
        'q5',
        "Beyond the governance of customer data used by AI features, what is Acme's framework for the responsible development of the AI models themselves? Specifically, what is your policy regarding the sourcing of training data for global models, and what governance processes are in place to validate models for fairness and bias before deployment?",
        ['ai_ethics_1', 'ai_governance_1'],
        {'ai_ethics_1': 1.0, 'ai_governance_1': 0.99}
    )
    
    return gt

# Add a function to run evaluation and return results in a serializable format
def run_evaluation() -> List[Dict]:
    """Run the full evaluation and return results in a JSON-serializable format"""
    
    # Step 1: Load and chunk document
    chunker = DocumentChunker("ACME Enterprise Platform.docx")
    chunks = chunker.load_and_chunk()
    
    # Step 2: Create ground truth
    ground_truth = create_ground_truth()
    
    # Step 3: Define models to evaluate
    models_to_evaluate = [
        "sentence-transformers/all-mpnet-base-v2",
        "BAAI/bge-large-en-v1.5",
        "intfloat/e5-large-v2"
    ]
    
    # Step 4: Evaluate each model
    all_results = []
    
    for model_name in models_to_evaluate:
        evaluator = EmbeddingModelEvaluator(model_name, chunks)
        results = evaluator.evaluate_on_dataset(ground_truth, top_k=3)
        all_results.append(results)
    
    # Convert results to serializable format
    serializable_results = []
    for result in all_results:
        model_data = {
            'model_name': result['model_name'],
            'average_metrics': asdict(result['average_metrics']),
            'per_question': {}
        }
        
        for q_id, q_data in result['per_question'].items():
            model_data['per_question'][q_id] = {
                'question': q_data['question'],
                'retrieved': [
                    {'chunk_id': r.chunk_id, 'score': r.score, 'rank': r.rank}
                    for r in q_data['retrieved']
                ],
                'metrics': asdict(q_data['metrics'])
            }
        
        serializable_results.append(model_data)
    
    return serializable_results

def print_summary_report(all_results: List[Dict]):
    """Print a comprehensive summary report"""
    
    print("\n" + "="*80)
    print("EMBEDDING MODEL EVALUATION - SUMMARY REPORT")
    print("="*80)
    
    # Create comparison table
    print("\n📊 OVERALL PERFORMANCE COMPARISON")
    print("-" * 80)
    
    df_data = []
    for result in all_results:
        metrics = result['average_metrics']
        df_data.append({
            'Model': result['model_name'],
            'Precision': f"{metrics['precision_at_k']:.3f}",
            'Recall': f"{metrics['recall_at_k']:.3f}",
            'F1': f"{metrics['f1_score']:.3f}",
            'MRR': f"{metrics['mrr']:.3f}",
            'NDCG': f"{metrics['ndcg']:.3f}",
            'MAP': f"{metrics['map_score']:.3f}"
        })
    
    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))
    
    # Find best model
    best_f1_idx = max(range(len(all_results)), 
                     key=lambda i: all_results[i]['average_metrics']['f1_score'])
    best_model = all_results[best_f1_idx]
    
    print("\n" + "="*80)
    print("🏆 RECOMMENDATION")
    print("="*80)
    print(f"\nBest Model: {best_model['model_name']}")
    print(f"\nKey Metrics:")
    metrics = best_model['average_metrics']
    print(f"  • F1 Score: {metrics['f1_score']:.3f}")
    print(f"  • MRR: {metrics['mrr']:.3f} (higher is better for RAG)")
    print(f"  • NDCG: {metrics['ndcg']:.3f}")
    print(f"  • Recall: {metrics['recall_at_k']:.3f}")
    
    print("\n📝 Rationale:")
    print("  • Highest overall F1 score indicates best balance of precision and recall")
    print("  • Strong MRR shows relevant results consistently appear at the top")
    print("  • High NDCG confirms quality ranking with proper relevance weighting")
    print("  • Excellent recall ensures all relevant information is retrieved")
    print("="*80 + "\n")

def main():
    """Main execution function"""
    
    print("="*80)
    print("EMBEDDING MODEL EVALUATION SYSTEM")
    print("="*80)
    
    # Run the evaluation
    results = run_evaluation()
    
    # Print summary report
    print_summary_report(results)
    
    print("\n✅ Evaluation complete!")
    print("="*80)

if __name__ == "__main__":
    main()