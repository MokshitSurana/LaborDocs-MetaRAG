"""
RAGAS Evaluation with Google Gemini - Legacy API
This version uses the old RAGAS API which is more stable with Gemini.
"""

import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import asyncio
import nest_asyncio

# Suppress deprecation warnings and runtime warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='coroutine.*was never awaited')
warnings.filterwarnings('ignore', message='Event loop is closed')

# Allow nested event loops (needed for RAGAS with Jupyter/async contexts)
nest_asyncio.apply()

# RAGAS Imports - Force legacy API
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    answer_correctness,
    answer_similarity
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# =============================================================================
# CONFIGURATION
# =============================================================================

# Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyDsifK7coXqATvA3BcrXCybt0fUejVdpNU")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

INPUT_DIRECTORY = r"C:\Users\moksh\Desktop\UICLaborDocsChatbot-clara_work\UICLaborDocsChatbot-clara_work\metarag\retrieval_output\run_1764103419\answers"

# Gemini models
EVAL_MODEL = "gemini-2.0-flash"  # More stable than 2.0 for RAGAS
EMBED_MODEL = "models/embedding-001"

# Increase timeouts to handle rate limiting
REQUEST_TIMEOUT = 180  # 3 minutes
MAX_RETRIES = 3

# =============================================================================
# TEST CASES
# =============================================================================

TEST_CASES = [
    {"id": 1, "question": "What FTE range must appointment fall within to be covered by the GEO contract?", "ground_truth": "Appointments must be at least 0.25 FTE (full-time equivalency) and no greater than 0.67 FTE to be covered by the GEO contract."},
    {"id": 2, "question": "How many days before the semester must appointment letters be issued?", "ground_truth": "Appointment letters shall be issued no later than 45 days before the start of the appointment. In cases where appointments are made less than 45 days before or after the beginning of the semester, the letter shall be issued as soon as practicable."},
    {"id": 3, "question": "Are tuition and fee waivers guaranteed for GEO-covered employees?", "ground_truth": "Yes, tuition waivers are guaranteed to teaching and graduate assistants holding a 0.25 to 0.67 FTE appointment during the term of the agreement. Once enrolled in a graduate program, students are governed by the tuition waiver policy in effect at the time of their first enrollment."},
    {"id": 4, "question": "What fees must the University waive for GEO members?", "ground_truth": "The University must waive the following fees: Service Fee, Health Service Fee, Academic Facilities and Maintenance Assessment, and Library and Information Technology Assessment. For international students, $65 of the International Student Fee is also waived for each Fall and Spring semester."},
    {"id": 5, "question": "What is the process for resolving grievances under the GEO collective bargaining agreement?", "ground_truth": "The grievance procedure has three levels before arbitration: Level 1 - Written grievance filed within 30 business days with the Unit Executive Officer (UEO), who must respond within 10 business days of the meeting. Level 2 - Appeal to the Dean of the College within 10 business days of Level 1 decision, with Dean responding within 10 business days of the meeting. Level 3 - Appeal to the Office of the Chancellor within 10 business days of Level 2 decision, with Chancellor responding within 10 business days of the meeting. If unresolved, the Union may submit to arbitration within 20 business days of the Level 3 decision."},
    {"id": 6, "question": "What are the minimum salaries for Assistant, Associate, and full Professors under the CBA?", "ground_truth": "Assistant Professor: $71,500; Associate Professor: $78,650; Professor: $86,515 (9-month salaries effective August 16, 2022)"},
    {"id": 7, "question": "How is merit pay determined and allocated for tenure-track faculty?", "ground_truth": "Merit-based salary increases are determined at the sole discretion of the department/dean following campus and unit policies. The University implements annual salary increase programs ranging from 2.5-3.5% of salary base, with individual faculty receiving increases that may be less than, equal to, or greater than the program percentage. Merit determinations are not grievable."},
    {"id": 8, "question": "What protections does the agreement provide regarding academic freedom?", "ground_truth": "The University maintains and encourages full freedom of inquiry, discourse, teaching, research, and publication within the law, protecting faculty from internal or external influences that would restrict these freedoms. Faculty who believe their academic freedom is violated may request a hearing before the Committee on Academic Freedom and Tenure. Substantive disputes are resolved under University Statutes (Article X, Section 2(d)), while procedural disputes may be grieved."},
    {"id": 9, "question": "What process governs non-reappointment or tenure decisions?", "ground_truth": "Appointments, reappointments, promotion, and tenure are made by the Board of Trustees following University of Illinois Statutes and campus policies. Tenure-track faculty enter a probationary period not exceeding seven years. Non-reappointment requires written notice at least 12 months before appointment expiration (or a terminal contract if less notice is given). These decisions are NOT subject to the grievance and arbitration procedure."},
    {"id": 10, "question": "What grievance and arbitration process applies to tenure-track faculty disputes?", "ground_truth": "Four-level process: Level 1 - immediate supervisor (14 days response); Level 2 - College Dean (14 days); Level 3 - Provost/VCHA (14 days); Level 4 - binding arbitration. Grievances must be filed within 25 business days of discovering the violation. The arbitrator is selected from FMCS panel and has authority limited to determining contract violations. Costs are split equally between parties. Decisions are binding."},
    {"id": 11, "question": "What is the minimum salary for instructors, lecturers, and clinical assistant professors?", "ground_truth": "$60,000 for full-time bargaining unit members with the rank of Instructor, Lecturer, Clinical Assistant Professor, Research Assistant Professor, or Teaching Assistant Professor"},
    {"id": 12, "question": "How are promotion and reappointment decisions handled for NTT faculty?", "ground_truth": "Procedures shall be provided in the bylaws of the bargaining unit faculty member's academic unit. If not included in bylaws, units must develop them within one year. Faculty with less than 10 years receive notice by May 15; faculty with 10+ years receive notice one year before contract end. These decisions are at the University's sole discretion and not grievable."},
    {"id": 13, "question": "What side letter provisions exist regarding teaching professor ranks?", "ground_truth": "No individual currently in a position modified by 'Clinical' or 'Research' will be required to convert to 'Teaching' modifier as a condition for reappointment. Units adopting the Teaching modifier must review workload expectations, evaluation criteria, and promotion norms within one year. Position changes require the faculty member to wish to be considered, possess the required degree, and have unit approval."},
    {"id": 14, "question": "Under what conditions can NTT faculty be laid off or recalled?", "ground_truth": "A bargaining unit member with a multi-year appointment may have their appointment terminated if the specific work they perform or courses they teach are reduced or discontinued. The affected faculty member may choose to be placed on a recall list for up to three academic years. If work is reinstated, faculty on the recall list shall be offered reinstatement to complete the remaining term of their multi-year appointment."},
    {"id": 15, "question": "What professional leave options are available to NTT faculty under the agreement?", "ground_truth": "The Office of the Provost will continue to work with the Faculty Senate to develop and implement a policy regarding the eligibility of NTT faculty to apply for professional leave, and shall report upon the progress toward this policy annually during the term of the contract."},
]

# =============================================================================
# FUNCTIONS
# =============================================================================

def setup_ragas():
    """Setup RAGAS with Gemini using legacy API."""
    try:
        print(f"Setting up RAGAS with {EVAL_MODEL}...")
        
        # Create Gemini LLM and embeddings with better error handling
        llm = ChatGoogleGenerativeAI(
            model=EVAL_MODEL,
            temperature=0,
            max_retries=MAX_RETRIES,
            timeout=REQUEST_TIMEOUT,
            request_timeout=REQUEST_TIMEOUT
        )
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBED_MODEL,
            task_type="retrieval_document",
            request_options={"timeout": REQUEST_TIMEOUT}
        )
        
        # Wrap for RAGAS
        ragas_llm = LangchainLLMWrapper(llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
        
        print("✓ RAGAS initialized")
        return ragas_llm, ragas_embeddings
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def load_json(filepath: Path):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_config(filepath: Path, ragas_llm, ragas_embeddings):
    """Evaluate a single configuration."""
    config_name = filepath.name.replace("_answers.json", "").replace(".json", "")
    print(f"\n{'='*70}\n📋 {config_name}\n{'='*70}")
    
    try:
        data = load_json(filepath)
    except Exception as e:
        print(f"❌ Error loading: {e}")
        return None

    # Handle different JSON structures
    queries = data.get("answers", data.get("queries", data))
    
    # Extract Q/A pairs
    q_list, a_list, c_list, gt_list = [], [], [], []
    
    for tc in TEST_CASES:
        q_key = f"q{tc['id']}"
        if q_key not in queries:
            continue
            
        item = queries[q_key]
        
        # Get answer
        answer = (
            item.get('answer') or 
            item.get('result') or 
            item.get('generated_answer') or 
            item.get('response')
        )
        
        if isinstance(answer, list) and answer:
            answer = answer[0]
        
        if not answer or not isinstance(answer, str):
            continue
            
        # Get contexts
        contexts = []
        if 'retrieved_chunks' in item:
            contexts = [c.get('text', c.get('page_content', '')) for c in item['retrieved_chunks']]
        elif 'source_documents' in item:
            contexts = [
                doc.get('page_content', '') if isinstance(doc, dict) else getattr(doc, 'page_content', '')
                for doc in item['source_documents']
            ]
        
        if not contexts:
            contexts = ["(No context available)"]
        
        contexts = [str(c) for c in contexts if c]
            
        q_list.append(tc['question'])
        a_list.append(answer)
        c_list.append(contexts)
        gt_list.append(tc['ground_truth'])

    if not q_list:
        print("⚠️  No valid Q/A pairs found")
        return None

    print(f"✓ Found {len(q_list)} Q/A pairs")
    print("⏳ Evaluating (this will take a few minutes)...")
    
    # Create dataset
    dataset = Dataset.from_dict({
        "question": q_list,
        "answer": a_list,
        "contexts": c_list,
        "ground_truth": gt_list
    })
    
    # Evaluate with retry and longer delays
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Add delay before evaluation to avoid rate limiting
            if attempt > 0:
                wait_time = 15 * attempt  # Progressive backoff
                print(f"   Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            
            results = evaluate(
                dataset=dataset,
                metrics=[
                    answer_relevancy,
                    faithfulness,
                    answer_correctness,
                    answer_similarity
                ],
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                raise_exceptions=False,
                show_progress=True
            )
            
            # Extract scores - handle both old dict and new EvaluationResult
            relevancy = 0
            faithfulness_score = 0
            correctness = 0
            similarity = 0
            
            if hasattr(results, 'to_pandas'):
                # New API
                df = results.to_pandas()
                # Check for NaN values and handle them
                relevancy = df['answer_relevancy'].mean()
                faithfulness_score = df['faithfulness'].mean()
                correctness = df['answer_correctness'].mean()
                similarity = df['answer_similarity'].mean()
                
                # If all NaN, the evaluation failed
                if pd.isna(relevancy) and pd.isna(faithfulness_score) and pd.isna(correctness) and pd.isna(similarity):
                    raise ValueError("All metrics returned NaN - evaluation failed")
                
                # Replace NaN with 0
                relevancy = relevancy if not pd.isna(relevancy) else 0
                faithfulness_score = faithfulness_score if not pd.isna(faithfulness_score) else 0
                correctness = correctness if not pd.isna(correctness) else 0
                similarity = similarity if not pd.isna(similarity) else 0
                
            elif isinstance(results, dict):
                # Old API (dict-like)
                relevancy = results.get("answer_relevancy", 0)
                faithfulness_score = results.get("faithfulness", 0)
                correctness = results.get("answer_correctness", 0)
                similarity = results.get("answer_similarity", 0)
            else:
                raise ValueError(f"Unexpected results type: {type(results)}")
            
            scores = {
                "Configuration": config_name,
                "Relevancy": round(relevancy, 3),
                "Faithfulness": round(faithfulness_score, 3),
                "Correctness": round(correctness, 3),
                "Similarity": round(similarity, 3)
            }
            
            print(f"✅ Success!")
            print(f"   Correctness: {scores['Correctness']:.3f}")
            print(f"   Faithfulness: {scores['Faithfulness']:.3f}")
            print(f"   Relevancy: {scores['Relevancy']:.3f}")
            
            return scores
            
        except Exception as e:
            if attempt < max_attempts - 1:
                print(f"⚠️  Attempt {attempt+1} failed: {e}")
                print("   Retrying in 10 seconds...")
                time.sleep(10)
            else:
                print(f"❌ Failed after {max_attempts} attempts: {e}")
                return None

def plot_results(df: pd.DataFrame, output_dir: Path):
    """Create visualization plots."""
    if df.empty:
        return
    
    metrics = ["Relevancy", "Faithfulness", "Correctness", "Similarity"]
    
    # Bar chart
    df_melted = df.melt(id_vars="Configuration", value_vars=metrics, var_name="Metric", value_name="Score")
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df_melted, x="Metric", y="Score", hue="Configuration", palette="viridis")
    plt.title("RAGAS Evaluation Results", fontsize=16, fontweight='bold')
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / "ragas_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Heatmap
    plt.figure(figsize=(10, max(6, len(df) * 0.5)))
    df_sorted = df.set_index("Configuration")[metrics].sort_values("Correctness", ascending=False)
    sns.heatmap(df_sorted, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1, linewidths=0.5)
    plt.title("Score Heatmap (Ranked by Correctness)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "ragas_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Plots saved to {output_dir}/")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("🚀 RAGAS EVALUATION WITH GOOGLE GEMINI")
    print("="*70)
    print(f"Input: {INPUT_DIRECTORY}")
    print(f"Model: {EVAL_MODEL}")
    print("="*70 + "\n")
    
    # Setup
    ragas_llm, ragas_embeddings = setup_ragas()
    if not ragas_llm:
        print("\n❌ Failed to initialize. Exiting.")
        return
    
    # Find files
    search_path = Path(INPUT_DIRECTORY)
    json_files = list(search_path.glob("*_answers.json"))
    
    if not json_files:
        print(f"\n❌ No *_answers.json files found in {INPUT_DIRECTORY}")
        return
    
    print(f"\n✓ Found {len(json_files)} file(s)\n")
    
    # Evaluate each
    all_scores = []
    for i, filepath in enumerate(json_files, 1):
        print(f"\n[{i}/{len(json_files)}] {filepath.name}")
        score = evaluate_config(filepath, ragas_llm, ragas_embeddings)
        if score:
            all_scores.append(score)
        
        # Rate limiting - wait between files to avoid quota issues
        if i < len(json_files):
            print("\n⏸️  Waiting 10 seconds (rate limiting)...")
            time.sleep(10)
            
            # Force garbage collection to clean up async resources
            import gc
            gc.collect()
    
    # Save results
    if not all_scores:
        print("\n❌ No results generated")
        return
    
    df = pd.DataFrame(all_scores)
    df_sorted = df.sort_values("Correctness", ascending=False)
    
    # Save CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = search_path / f"ragas_results_{timestamp}.csv"
    df_sorted.to_csv(csv_path, index=False)
    
    # Display
    print("\n" + "="*70)
    print("📊 FINAL RESULTS (Ranked by Correctness)")
    print("="*70)
    print(df_sorted.to_string(index=False))
    print("="*70)
    
    print(f"\n✅ Results saved to: {csv_path}")
    print(f"🏆 Best config: {df_sorted.iloc[0]['Configuration']}")
    print(f"   Correctness: {df_sorted.iloc[0]['Correctness']:.3f}")
    
    # Generate plots
    plot_results(df, search_path)
    
    print("\n✨ Evaluation complete!\n")

if __name__ == "__main__":
    main()