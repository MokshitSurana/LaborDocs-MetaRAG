#!/usr/bin/env python3
import os
import json
import time
import logging
import argparse
from typing import List, Dict, Any
import concurrent.futures
#from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAG-Answerer")

class LLMProcessor:
    """Class to handle LLM interactions."""
    
    ''' 
    # Azure OpenAI Deployment
    def __init__(self, model_name=None, temperature=0.5, max_retries=3, retry_delay=5):
        self.model_name = model_name or config.AZURE_DEPLOYMENT
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = None
        self.initialize_llm()
    '''

    # Hugging Face Deployment
    def __init__(self, model_name=None, temperature=0.5, max_retries=3, retry_delay=5):
        self.model_name = model_name or config.GEMINI_MODEL  
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = None
        self.initialize_llm()
    
    
    def initialize_llm(self):
        """Initialize the LLM client."""
        try:

            '''
            # Azure OpenAI client initialization
            self.client = AzureChatOpenAI(
                azure_deployment=config.AZURE_DEPLOYMENT,
                api_key=config.AZURE_API_KEY,
                api_version=config.AZURE_API_VERSION,
                azure_endpoint=config.AZURE_ENDPOINT,
                temperature=self.temperature
            )
            '''
            
            # Google Gemini with LangChain
            self.client = ChatGoogleGenerativeAI(
                model=config.GEMINI_MODEL,
                google_api_key=config.GOOGLE_API_KEY,
                temperature=self.temperature
            )


            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            self.client = None
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate an answer based on query and context."""
        if not self.client:
            logger.error("LLM client not initialized")
            return "Error: LLM not available"
        
        # Create system and user messages
        # Create system and user messages
        system_msg = """You are an expert on UIC labor agreements and employment contracts. 
        You specialize in analyzing collective bargaining agreements (CBAs).

        You will be provided with context chunks that contain two parts:
        1. **Metadata:** (Context Summaries, Document Types, Covered Groups)
        2. **Contract Excerpt:** (The raw text from the document)

        When answering:
        1. **Use information from BOTH the Metadata and the Contract Excerpt.** (e.g., If the text is a table of contents, but the metadata summary confirms the document covers 'wages', you may state that.)
        2. Quote specific numbers, percentages, and terms from the documents whenever possible.
        3. Cite which source document the information comes from (e.g., "[Source 1]").
        4. If the answer is not in the text OR the metadata, say "This information is not available in the provided documents".
        """

        user_msg = f"""Question: {query}

        Based on the following structured context (containing metadata summaries and contract text), provide a direct answer:

        {context}

        Answer: """
        
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg)
        ]
        
        # Try to generate with retries
        for attempt in range(self.max_retries):
            try:
                response = self.client.invoke(messages)
                return response.content
            except Exception as e:
                logger.error(f"Error generating answer (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
        
        return "Error: Failed to generate answer after multiple attempts"

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RAG Answer Generator")
    
    parser.add_argument("--retrieval_dir", type=str, required=True,
                       help="Directory containing retrieval outputs")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to store answer outputs")
    parser.add_argument("--top_k", type=int, default=10,
                       help="Number of top chunks to use for context")
    parser.add_argument("--threads", type=int, default=4,
                       help="Number of parallel threads")
    parser.add_argument("--rate_limit", type=float, default=10,
                       help="Maximum requests per minute")
    
    return parser.parse_args()

def load_retrieval_files(retrieval_dir: str) -> List[str]:
    """Load all retrieval result files."""
    retrieval_files = []
    
    for filename in os.listdir(retrieval_dir):
        if filename.endswith("_retrieval.json"):
            retrieval_files.append(os.path.join(retrieval_dir, filename))
    
    return retrieval_files


def process_retrieval_file(file_path: str, top_k: int, llm_processor: LLMProcessor) -> Dict[str, Any]:
    """Process a single retrieval file."""
    logger.info(f"Processing {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        retriever_name = data.get("run_info", {}).get("retriever_name", "unknown")
        answers = {}
        
        for query_id, query_data in data.get("queries", {}).items():
            query_text = query_data.get("query_text", "")
            chunks = query_data.get("retrieved_chunks", [])
            
            logger.info(f"Query {query_id}: {len(chunks)} chunks available in file")
            
            # Count chunks with text
            chunks_with_text = sum(1 for chunk in chunks if chunk.get("text", "").strip())
            logger.info(f"Query {query_id}: {chunks_with_text} chunks have non-empty text")
            
            # Extract text AND metadata from top-k chunks
            context_parts = []
            for i, chunk in enumerate(chunks[:top_k], 1):
                chunk_text = chunk.get("text", "").strip()
                
                if chunk_text:
                    # 1. Get Basic Info
                    doc_name = chunk.get("document_name", "Unknown Document")
                    
                    # 2. Extract valuable metadata
                    # We use .get() to be safe if keys are missing
                    summary = chunk.get("summary", "")
                    content_type = chunk.get("content_type", "")
                    employee_group = chunk.get("covered_employee_group", "")
                    
                    # 3. Build a metadata header string
                    meta_header = []
                    if content_type:
                        meta_header.append(f"Type: {content_type}")
                    if employee_group and employee_group != "Not Specified":
                        meta_header.append(f"Applies to: {employee_group}")
                    if summary:
                        meta_header.append(f"Context Summary: {summary}")
                    
                    meta_str = "\n".join(meta_header)
                    
                    # 4. Construct the rich context block
                    # We clearly separate Metadata from Content so the LLM doesn't get confused
                    block = (
                        f"[Source {i}: {doc_name}]\n"
                        f"{meta_str}\n"
                        f"--- Contract Excerpt ---\n"
                        f"{chunk_text}"
                    )
                    
                    context_parts.append(block)

            context = "\n\n====================\n\n".join(context_parts)
            
            logger.info(f"Query {query_id}: Using {len(context_parts)} chunks out of {len(chunks)} available")
            
            # Generate answer
            answer = llm_processor.generate_answer(query_text, context)
            
            # Store result
            source_docs = sorted(set(
                chunk.get("document_name", "Unknown Document")
                for chunk in chunks[:top_k]
                if chunk.get("document_name")
            ))

            answers[query_id] = {
                "query": query_text,
                "answer": answer,
                "num_chunks_used": len(context_parts),
                "sources": source_docs,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        
        return {
            "retriever_name": retriever_name,
            "answers": answers
        }
    
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return {
            "retriever_name": os.path.basename(file_path),
            "error": str(e),
            "answers": {}
        }

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(args.retrieval_dir, "answers")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize LLM processor
    llm_processor = LLMProcessor()
    
    # Load retrieval files
    retrieval_files = load_retrieval_files(args.retrieval_dir)
    
    if not retrieval_files:
        logger.error(f"No retrieval files found in {args.retrieval_dir}")
        return
    
    # Calculate rate limit delay
    rate_limit_delay = 60.0 / args.rate_limit
    
    # Process files with thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = []
        
        for file_path in retrieval_files:
            future = executor.submit(process_retrieval_file, file_path, args.top_k, llm_processor)
            futures.append(future)
            # Apply rate limiting
            time.sleep(rate_limit_delay)
        
        # Collect results
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                retriever_name = result.get("retriever_name", "unknown")
                
                # Save individual result
                sanitized_name = retriever_name.replace(" ", "_").replace("(", "_").replace(")", "_")
                output_path = os.path.join(args.output_dir, f"{sanitized_name}_answers.json")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2)
                
                logger.info(f"Saved answers for {retriever_name} to {output_path}")
            
            except Exception as e:
                logger.error(f"Error processing result: {str(e)}")
    
    # Create summary file
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "retriever_count": len(retrieval_files),
        "top_k": args.top_k,
        "output_dir": args.output_dir
    }
    
    summary_path = os.path.join(args.output_dir, "answers_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Answer generation completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()