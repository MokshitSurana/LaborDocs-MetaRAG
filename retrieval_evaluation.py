# retrieval_evaluation.py
"""
Hybrid Retriever Evaluation Script
Evaluates the QUALITY of retrieved chunks, not the final LLM answer

Metrics:
1. Precision@K - How many retrieved docs are relevant?
2. Recall@K - Did we find all relevant docs?
3. MRR (Mean Reciprocal Rank) - Where is the first relevant doc?
4. NDCG@K - Normalized Discounted Cumulative Gain
5. Hit Rate@K - Did we find at least one relevant doc?
6. Context Precision - Are retrieved chunks useful for the question?
7. Context Recall - Do we have enough info to answer?
"""

import json
import os
import glob
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import numpy as np
import pandas as pd  # Optional, for pretty printing tables
import matplotlib.pyplot as plt
import seaborn as sns
# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to your run folder
INPUT_DIRECTORY = r"C:\Users\moksh\Desktop\UICLaborDocsChatbot-clara_work\UICLaborDocsChatbot-clara_work\metarag\retrieval_output\run_1764103419"

# EVALUATION SETTING: Top-K to evaluate
EVAL_K = 10


# =============================================================================
# TEST CASES WITH GROUND TRUTH
# =============================================================================

TEST_CASES = [
   {
        "id": 1,
        "question": "What FTE range must appointment fall within to be covered by the GEO contract?",
        "expected_answer": "at least 0.25 FTE and no greater than 0.67 FTE",
        "relevant_keywords": ["fte", "0.25", ".25", "0.67", ".67", "full-time", "equivalency", "bargaining unit"],
        "relevant_sources": ["GEO"],
        "must_contain": ["0.25", "0.67"],
        # "category": "factual"
    },
    {
        "id": 2,
        "question": "How many days before the semester must appointment letters be issued?",
        "expected_answer": "45 days before the start of the appointment. In cases of appointments made less than 45 days before or after the beginning of the semester, the letter shall be issued as soon as practicable.",
        "relevant_keywords": ["45", "days", "appointment", "letter", "semester", "issued", "before"],
        "relevant_sources": ["GEO"],
        "must_contain": ["45"],
        # "category": "factual"
    },
    {
        "id": 3,
        "question": "Are tuition and fee waivers guaranteed for GEO-covered employees?",
        "expected_answer": "Yes, tuition waivers are guaranteed to teaching and graduate assistants holding a 0.25 to 0.67 FTE appointment. Assistants covered by this Agreement have graduate assistant appointments which are no less than 0.25 FTE and no greater than 0.67 FTE, or who otherwise are granted a tuition waiver.",
        "relevant_keywords": ["tuition", "waiver", "guaranteed", "fte", "appointment", "teaching", "graduate"],
        "relevant_sources": ["GEO"],
        "must_contain": ["tuition", "waiver"],
        # "category": "factual"
    },
    {
        "id": 4,
        "question": "What fees must the University waive for GEO members (e.g., service fee, health service fee)?",
        "expected_answer": "Service Fee, Health Service Fee, Academic Facilities and Maintenance Assessment, and Library and Information Technology Assessment. Additionally, for International students, the University shall waive $65 of the International Student Fee for each Fall and Spring semester.",
        "relevant_keywords": ["fee", "waive", "service", "health", "academic", "facilities", "library", "assessment"],
        "relevant_sources": ["GEO"],
        "must_contain": ["Service Fee", "Health Service Fee"],
        # "category": "factual"
    },
    {
        "id": 5,
        "question": "What is the process for resolving grievances under the GEO collective bargaining agreement?",
        "expected_answer": "The grievance procedure has three levels before arbitration: Level 1 - Written grievance filed within 30 business days with the Unit Executive Officer (UEO), who must respond within 10 business days of the meeting. Level 2 - Appeal to the Dean of the College within 10 business days of Level 1 decision, with Dean responding within 10 business days of the meeting. Level 3 - Appeal to the Office of the Chancellor within 10 business days of Level 2 decision, with Chancellor responding within 10 business days of the meeting. If unresolved, the Union may submit to arbitration within 20 business days of the Level 3 decision. The arbitrator is selected through FMCS if parties cannot agree, and their decision is final and binding.",
        "relevant_keywords": ["grievance", "level", "appeal", "arbitration", "UEO", "dean", "chancellor", "30 business days", "10 business days", "written"],
        "relevant_sources": ["GEO"],
        "must_contain": ["Level", "arbitration", "30 business days"],
    },
    {
        "id": 6,
        "question": "What are the minimum salaries for Assistant, Associate, and full Professors under the CBA?",
        "expected_answer": "Assistant Professor: $71,500; Associate Professor: $78,650; Professor: $86,515 (9-month salaries effective August 16, 2022)",
        "relevant_keywords": ["minimum", "salary", "assistant", "associate", "professor", "71500", "78650", "86515"],
        "relevant_sources": ["TT-FACULTY"],
        "must_contain": ["71,500", "78,650", "86,515"],
        # "category": "factual"
    },
    {
        "id": 7,
        "question": "How is merit pay determined and allocated for tenure-track faculty?",
        "expected_answer": "Merit-based salary increases are determined at the sole discretion of the department/dean following campus and unit policies. The University implements annual salary increase programs ranging from 2.5-3.5% of salary base, with individual faculty receiving increases that may be less than, equal to, or greater than the program percentage. Merit determinations are not grievable.",
        "relevant_keywords": ["merit", "salary", "increase", "discretion", "dean", "department", "not grievable", "3.5%", "3%"],
        "relevant_sources": ["TT-FACULTY"],
        "must_contain": ["merit", "discretion", "not grievable"],
        # "category": "process"
    },
    {
        "id": 8,
        "question": "What protections does the agreement provide regarding academic freedom?",
        "expected_answer": "The University maintains and encourages full freedom of inquiry, discourse, teaching, research, and publication within the law, protecting faculty from internal or external influences that would restrict these freedoms. Faculty who believe their academic freedom is violated may request a hearing before the Committee on Academic Freedom and Tenure. Substantive disputes are resolved under University Statutes (Article X, Section 2(d)), while procedural disputes may be grieved.",
        "relevant_keywords": ["academic freedom", "inquiry", "teaching", "research", "publication", "committee", "hearing", "protection"],
        "relevant_sources": ["TT-FACULTY"],
        "must_contain": ["academic freedom", "inquiry", "research"],
        # "category": "rights"
    },
    {
        "id": 9,
        "question": "What process governs non-reappointment or tenure decisions?",
        "expected_answer": "Appointments, reappointments, promotion, and tenure are made by the Board of Trustees following University of Illinois Statutes and campus policies. Tenure-track faculty enter a probationary period not exceeding seven years. Non-reappointment requires written notice at least 12 months before appointment expiration (or a terminal contract if less notice is given). These decisions are NOT subject to the grievance and arbitration procedure.",
        "relevant_keywords": ["non-reappointment", "tenure", "Board of Trustees", "probationary", "seven years", "12 months", "notice", "not subject to grievance"],
        "relevant_sources": ["TT-FACULTY"],
        "must_contain": ["Board of Trustees", "not subject", "grievance"],
        # "category": "process"
    },
    {
        "id": 10,
        "question": "What grievance and arbitration process applies to tenure-track faculty disputes?",
        "expected_answer": "Four-level process: Level 1 - immediate supervisor (14 days response); Level 2 - College Dean (14 days); Level 3 - Provost/VCHA (14 days); Level 4 - binding arbitration. Grievances must be filed within 25 business days of discovering the violation. The arbitrator is selected from FMCS panel and has authority limited to determining contract violations. Costs are split equally between parties. Decisions are binding.",
        "relevant_keywords": ["grievance", "arbitration", "four levels", "25 business days", "supervisor", "dean", "provost", "arbitrator", "binding"],
        "relevant_sources": ["TT-FACULTY"],
        "must_contain": ["25 business days", "arbitration", "binding"],
        # "category": "process"
    },

    {
        "id": 11,
        "question": "What is the minimum salary for instructors, lecturers, and clinical assistant professors?",
        "expected_answer": "$60,000 for full-time bargaining unit members with the rank of Instructor, Lecturer, Clinical Assistant Professor, Research Assistant Professor, or Teaching Assistant Professor",
        "relevant_keywords": ["minimum", "salary", "instructor", "lecturer", "clinical assistant professor", "$60,000", "60000", "full-time"],
        "relevant_sources": ["NTT-FACULTY"],
        "must_contain": ["60,000", "Instructor"],
        # "category": "factual"
    },
    {
        "id": 12,
        "question": "How are promotion and reappointment decisions handled for NTT faculty?",
        "expected_answer": "Procedures shall be provided in the bylaws of the bargaining unit faculty member's academic unit. If not included in bylaws, units must develop them within one year. Faculty with less than 10 years receive notice by May 15; faculty with 10+ years receive notice one year before contract end. These decisions are at the University's sole discretion and not grievable.",
        "relevant_keywords": ["promotion", "reappointment", "bylaws", "May 15", "ten years", "discretion", "grievable", "academic unit"],
        "relevant_sources": ["NTT-FACULTY"],
        "must_contain": ["bylaws", "May 15", "discretion"],
        # "category": "procedural"
    },
    {
        "id": 13,
        "question": "What side letter provisions exist regarding teaching professor ranks?",
        "expected_answer": "No individual currently in a position modified by 'Clinical' or 'Research' will be required to convert to 'Teaching' modifier as a condition for reappointment. Units adopting the Teaching modifier must review workload expectations, evaluation criteria, and promotion norms within one year. Position changes require the faculty member to wish to be considered, possess the required degree, and have unit approval.",
        "relevant_keywords": ["teaching professor", "side letter", "clinical", "research", "modifier", "convert", "voluntary", "workload", "promotion norms"],
        "relevant_sources": ["NTT-FACULTY"],
        "must_contain": ["Teaching", "required", "convert"],
        # "category": "policy"
    },
    {
        "id": 14,
        "question": "Under what conditions can NTT faculty be laid off or recalled?",
        "expected_answer": "A bargaining unit member with a multi-year appointment may have their appointment terminated if the specific work they perform or courses they teach are reduced or discontinued. The affected faculty member may choose to be placed on a recall list for up to three academic years. If work is reinstated, faculty on the recall list shall be offered reinstatement to complete the remaining term of their multi-year appointment.",
        "relevant_keywords": ["layoff", "recall", "multi-year", "terminated", "discontinued", "three years", "reinstatement", "recall list"],
        "relevant_sources": ["NTT-FACULTY"],
        "must_contain": ["multi-year", "recall", "three"],
        # "category": "procedural"
    },
    {
        "id": 15,
        "question": "What professional leave options are available to NTT faculty under the agreement?",
        "expected_answer": "The Office of the Provost will continue to work with the Faculty Senate to develop and implement a policy regarding the eligibility of NTT faculty to apply for professional leave, and shall report upon the progress toward this policy annually during the term of the contract.",
        "relevant_keywords": ["professional leave", "side letter", "Provost", "Faculty Senate", "develop", "policy", "eligibility", "annually"],
        "relevant_sources": ["NTT-FACULTY"],
        "must_contain": ["professional leave", "develop", "policy"],
        # "category": "benefits"
    },
]



# =============================================================================
# HELPER CLASSES
# =============================================================================

class MockDocument:
    """Mocks the LangChain Document object."""
    def __init__(self, text: str, metadata: Dict):
        self.page_content = text
        self.metadata = metadata

class RetrievalEvaluator:
    """Evaluates retrieval quality."""
    
    def precision_at_k(self, retrieved_docs: List, relevant_keywords: List[str], k: int = 10) -> float:
        relevant_count = 0
        # Slice top-k
        docs_to_check = retrieved_docs[:k]
        
        for doc in docs_to_check:
            content_lower = doc.page_content.lower()
            if any(keyword.lower() in content_lower for keyword in relevant_keywords):
                relevant_count += 1
        
        # Precision = (Relevant Retrieved) / K
        return relevant_count / k if k > 0 else 0.0
    
    def recall_at_k(self, retrieved_docs: List, must_contain: List[str], k: int = 10) -> float:
        docs_to_check = retrieved_docs[:k]
        retrieved_text = " ".join([doc.page_content.lower() for doc in docs_to_check])
        
        if not must_contain: return 1.0
        
        found_count = sum(1 for term in must_contain if term.lower() in retrieved_text)
        return found_count / len(must_contain)
    
    def mrr(self, retrieved_docs: List, relevant_keywords: List[str]) -> float:
        # MRR looks at the ENTIRE retrieved list to find the FIRST relevant match
        for rank, doc in enumerate(retrieved_docs, start=1):
            content_lower = doc.page_content.lower()
            if any(keyword.lower() in content_lower for keyword in relevant_keywords):
                return 1.0 / rank
        return 0.0
    
    def hit_rate_at_k(self, retrieved_docs: List, relevant_keywords: List[str], k: int = 10) -> float:
        docs_to_check = retrieved_docs[:k]
        for doc in docs_to_check:
            content_lower = doc.page_content.lower()
            if any(keyword.lower() in content_lower for keyword in relevant_keywords):
                return 1.0
        return 0.0
    
    def source_accuracy(self, retrieved_docs: List, expected_sources: List[str]) -> float:
        if not retrieved_docs: return 0.0
        correct_count = 0
        for doc in retrieved_docs:
            # Handle cases where source might be full path or just filename
            source_meta = doc.metadata.get("document_name", "") or doc.metadata.get("source", "")
            source_meta = source_meta.upper()
            
            if any(exp_src.upper() in source_meta for exp_src in expected_sources):
                correct_count += 1
        return correct_count / len(retrieved_docs)

# =============================================================================
# PROCESSING LOGIC
# =============================================================================

def load_retrieval_data(filepath: str) -> Dict:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def convert_to_docs(json_chunks: List[Dict]) -> List[MockDocument]:
    """Converts JSON chunk dicts to MockDocument objects."""
    docs = []
    for chunk in json_chunks:
        # Support both 'text' (retrieval.json) and 'page_content' keys
        text_content = chunk.get('text', chunk.get('page_content', ''))
        
        docs.append(MockDocument(
            text=text_content,
            metadata={
                'document_name': chunk.get('document_name', ''),
                'chunk_id': chunk.get('chunk_id', ''),
                'score': chunk.get('score', 0)
            }
        ))
    return docs

def evaluate_single_file(filepath: Path, evaluator: RetrievalEvaluator, k: int) -> Dict:
    """Runs evaluation metrics on a single JSON file using top-K."""
    try:
        data = load_retrieval_data(filepath)
    except json.JSONDecodeError:
        print(f"Error decoding {filepath.name}")
        return None

    # The structure of *_retrieval.json is typically {"queries": {"q1": ...}}
    queries = data.get('queries', {})
    
    if not queries and "q1" in data: 
        # Handle edge case where json is just {"q1": [...]} (results.json format)
        queries = data 

    aggregate_metrics = {
        f"precision@{k}": [],
        f"recall@{k}": [],
        "mrr": [],
        f"hit_rate@{k}": [],
        "source_accuracy": []
    }

    # Iterate over our Ground Truth Test Cases
    for test_case in TEST_CASES:
        # Construct query key (assuming format "q1", "q2"...)
        q_id_key = f"q{test_case['id']}"
        
        retrieved_chunks = []
        
        # Extract chunks based on JSON structure
        if q_id_key in queries:
            q_data = queries[q_id_key]
            if isinstance(q_data, dict) and 'retrieved_chunks' in q_data:
                retrieved_chunks = q_data['retrieved_chunks']
            elif isinstance(q_data, list):
                retrieved_chunks = q_data
        
        if not retrieved_chunks:
            # Zero out if no chunks found for this query
            aggregate_metrics[f'precision@{k}'].append(0.0)
            aggregate_metrics[f'recall@{k}'].append(0.0)
            aggregate_metrics['mrr'].append(0.0)
            aggregate_metrics[f'hit_rate@{k}'].append(0.0)
            aggregate_metrics['source_accuracy'].append(0.0)
            continue
        
        # Convert to doc objects
        retrieved_docs = convert_to_docs(retrieved_chunks)
        
        # Calculate Metrics using K
        aggregate_metrics[f'precision@{k}'].append(
            evaluator.precision_at_k(retrieved_docs, test_case['relevant_keywords'], k=k)
        )
        aggregate_metrics[f'recall@{k}'].append(
            evaluator.recall_at_k(retrieved_docs, test_case['must_contain'], k=k)
        )
        aggregate_metrics['mrr'].append(
            evaluator.mrr(retrieved_docs, test_case['relevant_keywords'])
        )
        aggregate_metrics[f'hit_rate@{k}'].append(
            evaluator.hit_rate_at_k(retrieved_docs, test_case['relevant_keywords'], k=k)
        )
        aggregate_metrics['source_accuracy'].append(
            evaluator.source_accuracy(retrieved_docs, test_case['relevant_sources'])
        )

    # Average stats for this file
    final_stats = {
        "filename": filepath.name.replace("_retrieval.json", "").replace(".json", ""),
        f"precision@{k}": np.mean(aggregate_metrics[f'precision@{k}']),
        f"recall@{k}": np.mean(aggregate_metrics[f'recall@{k}']),
        "mrr": np.mean(aggregate_metrics['mrr']),
        f"hit_rate@{k}": np.mean(aggregate_metrics[f'hit_rate@{k}']),
        "source_acc": np.mean(aggregate_metrics['source_accuracy']),
    }
    
    return final_stats

def plot_results(results: List[Dict], output_dir: Path):
    """Generates comparison plots."""
    df = pd.DataFrame(results)
    
    # These exact keys must match the return dict of evaluate_single_file
    metrics_to_plot = ["mrr", f"hit_rate@{EVAL_K}", f"precision@{EVAL_K}", f"recall@{EVAL_K}"]
    
    # Melt
    df_melted = df.melt(id_vars="filename", value_vars=metrics_to_plot, var_name="Metric", value_name="Score")
    
    # Plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 8))
    
    chart = sns.barplot(
        data=df_melted,
        x="Metric",
        y="Score",
        hue="filename",
        palette="viridis",
        alpha=0.9
    )
    
    plt.title(f"Retrieval Performance Comparison (Top-{EVAL_K})", fontsize=16, pad=20)
    plt.ylim(0, 1.05) # slightly above 1.0 for visual room
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    
    plot_path = output_dir / "retrieval_comparison_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\n📊 Plot saved to: {plot_path}")
    
    # Heatmap
    plt.figure(figsize=(10, len(df) * 0.5 + 2))
    df_indexed = df.set_index("filename")[metrics_to_plot].sort_values("mrr", ascending=False)
    sns.heatmap(df_indexed, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=.5)
    plt.title("Leaderboard (Heatmap)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "retrieval_leaderboard.png", dpi=300)
# =============================================================================
# MAIN RUNNER
# =============================================================================

if __name__ == "__main__":
    evaluator = RetrievalEvaluator()
    search_path = Path(INPUT_DIRECTORY)
    json_files = list(search_path.glob("*_retrieval.json"))
    
    if not json_files:
        print(f"❌ No *_retrieval.json files found in {INPUT_DIRECTORY}")
        exit()
        
    print(f"Found {len(json_files)} configurations. Evaluating...")
    
    all_results = []
    for json_file in json_files:
        try:
            stats = evaluate_single_file(json_file, evaluator, k=EVAL_K)
            if stats: all_results.append(stats)
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")

    if all_results:
        # Sort for terminal display
        all_results.sort(key=lambda x: x['mrr'], reverse=True)
        
        # Print Leaderboard
        print("\n" + "="*110)
        print(f"{'CONFIGURATION':<45} | MRR    | HIT@{EVAL_K}  | PREC@{EVAL_K} | REC@{EVAL_K}  | SRC %")
        print("="*110)
        
        for res in all_results:
            print(f"{res['filename'][:45]:<45} | {res['mrr']:.3f}  | {res[f'hit_rate@{EVAL_K}']:.3f}   | {res[f'precision@{EVAL_K}']:.3f}   | {res[f'recall@{EVAL_K}']:.3f}   | {res['source_acc']:.3f}")
        print("="*110)

        # Save CSV
        output_csv = search_path / f"evaluation_summary_k{EVAL_K}.csv"
        pd.DataFrame(all_results).to_csv(output_csv, index=False)
        print(f"\n✅ CSV summary saved to: {output_csv}")
        
        # Plot
        plot_results(all_results, search_path)
    else:
        print("No valid results generated.")