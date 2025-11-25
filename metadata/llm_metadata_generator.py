import json
import time
import os
from typing import Dict, List, Any, Optional
import random
#from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import re
import config
from metadata.base_metadata_generator import BaseMetadataGenerator

from gpu_utils import GPUVerifier

# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)

class LLMMetadataGenerator(BaseMetadataGenerator):
    """Metadata generator using LLM."""
    
    def __init__(self, output_dir="metadata_gen_output"):
        """Initialize the LLM metadata generator."""
        super().__init__(output_dir)
        
        # Initialize LLM
        try:

            '''
            # Azure OpenAI
            #self.client = AzureChatOpenAI(
            #    azure_deployment=config.AZURE_DEPLOYMENT,
            #    api_key=config.AZURE_API_KEY,
            #    api_version=config.AZURE_API_VERSION,
            #    azure_endpoint=config.AZURE_ENDPOINT,
            #    temperature=config.TEMPERATURE
            )
            '''

            # Google Gemini with LangChain
            self.client = ChatGoogleGenerativeAI(
                model=config.GEMINI_MODEL,
                google_api_key=config.GOOGLE_API_KEY,
                temperature=config.TEMPERATURE
            )

            self.logger.info("LLM initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {str(e)}")
            self.client = None
    

    def _enrich_chunks(self, chunk_data, method):
        """Enrich chunks with metadata using LLM, skipping identical documents or chunks intelligently."""
        if not self.client:
            self.logger.error("LLM not initialized, cannot enrich chunks")
            return chunk_data

        import hashlib, uuid, os, time, random, json

        chunks = chunk_data.get("chunks", [])
        document_name = chunk_data.get("document_name", "unknown")

        # --- Step 0: Generate a stable document_id (based on file content if possible) ---
        raw_id = chunk_data.get("document_id")
        if not raw_id or len(raw_id) < 8:
            possible_file = os.path.join("input_files", document_name)
            if os.path.exists(possible_file):
                with open(possible_file, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
                document_id = file_hash
                self.logger.info(f"Using stable document_id from file hash: {document_id}")
            else:
                document_id = str(uuid.uuid4())
                self.logger.warning(f"Could not find file for {document_name}, using random ID: {document_id}")
        else:
            document_id = raw_id

        self.logger.info(f"Starting enrichment for document: {document_name} ({len(chunks)} chunks)")

        # --- Step 1: Check if this entire document is already enriched ---
        output_subdir = f"{method}_chunks_metadata"
        enriched_file_path = os.path.join(
            self.output_dir, output_subdir, f"{document_name.split('.')[0]}_enriched_chunks.json"
        )

        if os.path.exists(enriched_file_path):
            try:
                with open(enriched_file_path, 'r', encoding='utf-8') as ef:
                    existing_data = json.load(ef)
                    existing_doc_id = existing_data.get("metadata", {}).get("enrichment", {}).get("document_id")
                    if existing_doc_id == document_id:
                        self.logger.info(f"Skipping document: {document_name} — already enriched with same document_id.")
                        return existing_data
            except Exception as e:
                self.logger.warning(f"Could not check existing enriched file: {e}")

        # --- Step 2: Load previous enriched chunks for chunk-level skipping ---
        existing_chunks_map = {}
        if os.path.exists(enriched_file_path):
            try:
                with open(enriched_file_path, 'r', encoding='utf-8') as ef:
                    existing_data = json.load(ef)
                    for ec in existing_data.get("chunks", []):
                        text = ec.get("text", "")
                        text_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
                        sig = f"{ec.get('chunk_id', '')}::{text_hash}"
                        existing_chunks_map[sig] = ec
                self.logger.info(f"Loaded {len(existing_chunks_map)} existing chunks for partial skipping.")
            except Exception as e:
                self.logger.warning(f"Could not read existing metadata file: {e}")

        # --- Step 3: Process chunks in batches ---
        enriched_chunks = []
        batch_size = config.BATCH_SIZE

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.logger.info(f"Processing batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}")

            for chunk in batch:
                chunk_id = chunk.get("chunk_id", "")
                text = chunk.get("text", "")
                text_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
                signature = f"{chunk_id}::{text_hash}"

                # --- Smart skip check ---
                if signature in existing_chunks_map:
                    self.logger.info(f"Skipping identical chunk: {chunk_id}")
                    enriched_chunks.append(existing_chunks_map[signature])
                    continue

                try:
                    enriched_chunk = self._enrich_single_chunk(chunk, method)
                    enriched_chunk["metadata"]["_signature"] = signature
                    enriched_chunks.append(enriched_chunk)
                    self.logger.info(f"Enriched new/updated chunk: {chunk_id}")
                except Exception as e:
                    self.logger.error(f"Error enriching chunk {chunk_id}: {str(e)}")
                    enriched_chunks.append(chunk)

                # --- Delay to avoid API rate limits ---
                time.sleep(random.uniform(0.5, 1.5))

            if i + batch_size < len(chunks):
                self.logger.info("⏸️ Pausing briefly between batches to avoid rate limits.")
                time.sleep(random.uniform(1.0, 3.0))

        # --- Step 4: Wrap up results ---
        result = chunk_data.copy()
        result["chunks"] = enriched_chunks
        result["metadata"]["enrichment"] = {
            "method": "llm",
            "model": config.GEMINI_MODEL,
            "enriched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "document_id": document_id
        }

        skipped_count = sum(
            1 for c in enriched_chunks if c.get("metadata", {}).get("_signature") in existing_chunks_map
        )
        new_count = len(chunks) - skipped_count
        self.logger.info(f"Summary for {document_name}: {skipped_count} skipped, {new_count} newly enriched chunks.")
        self.logger.info(f"Finished enrichment for {document_name} — total: {len(enriched_chunks)} chunks")

        return result
    
    def _enrich_single_chunk(self, chunk, method):
        """Enrich a single chunk with metadata."""
        # Extract text content
        text = chunk.get("text", "")
        
        # Generate all metadata using a single LLM call
        metadata = self._generate_combined_metadata(text)
        
        # Create enhanced embedding data
        embedding_enhancement = self._generate_embedding_enhancement(
            text, 
            metadata.get("content", {}), 
            metadata.get("contract", {}), 
            metadata.get("semantic", {})
        )
        
        # Create enriched chunk
        enriched_chunk = chunk.copy()
        
        # Remove redundant metadata if present
        if "metadata" in enriched_chunk:
            existing_metadata = enriched_chunk["metadata"]
            # Remove metadata that won't help with retrieval
            keys_to_remove = ["page_range", "code_lines", "processing_time"]
            for key in keys_to_remove:
                if key in existing_metadata:
                    del existing_metadata[key]
        else:
            enriched_chunk["metadata"] = {}
        
        # Add new metadata
        enriched_chunk["metadata"]["content"] = metadata.get("content", {})
        enriched_chunk["metadata"]["contract"] = metadata.get("contract", {})
        enriched_chunk["metadata"]["semantic"] = metadata.get("semantic", {})
        enriched_chunk["embedding_enhancement"] = embedding_enhancement
        
        return enriched_chunk 

    def _clean_json_string(self, json_string):
        """Helper function to clean and fix common JSON errors."""
        # Remove code block markers
        if "```json" in json_string:
            parts = json_string.split("```json", 1)
            if len(parts) > 1:
                json_string = parts[1]
            if "```" in json_string:
                json_string = json_string.split("```", 1)[0]
        
        # Trim whitespace
        json_string = json_string.strip()
        
        # Fix common JSON errors - extra commas before closing brackets
        json_string = re.sub(r',\s*}', '}', json_string)
        json_string = re.sub(r',\s*]', ']', json_string)
        
        return json_string


    def _parse_json_safely(self, content):
        """Safely parse JSON, handling errors and fixing common issues."""
        try:
            # Try direct parsing first
            return json.loads(content)
        except json.JSONDecodeError:
            # Clean up the JSON string
            cleaned_json = self._clean_json_string(content)
            
            try:
                # Try parsing the cleaned JSON
                return json.loads(cleaned_json)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON parsing error after cleanup: {str(e)}")
                self.logger.error(f"Cleaned content: {cleaned_json}")
                raise


    def _generate_combined_metadata(self, text):
        """Generate all metadata using a single LLM call for union labor agreements."""
        try:
            prompt = f"""You are analyzing a section from a union collective bargaining agreement or supplemental wage agreement between the University of Illinois Chicago and various labor unions. Extract structured metadata to help HR staff, union representatives, and employees find relevant information.

            
            TEXT:
            {text}
            
            OUTPUT JSON with these fields:

            1. content: object with:
            - content_type: object with "primary" and "subtypes" array
                PRIMARY options (choose ONE):
                * "Wage_Table" - salary scales, hourly rates, step increases
                * "Contract_Provisions" - rules, policies, procedures in text form
                * "Benefits" - health insurance, leave, retirement details
                * "Working_Conditions" - hours, shifts, schedules, safety
                * "Employment_Terms" - hiring, probation, appointments, promotions
                * "Rights_and_Protections" - non-discrimination, union rights, academic freedom
                * "Dispute_Resolution" - grievance procedures, arbitration
                * "Administrative" - recognition, dues, no-strike clauses, duration
                
                SUBTYPES: more specific topics (max 3)
                Examples: ["Overtime Rules", "Vacation Accrual", "Seniority Rights"]
            
            - keywords: array of important terms (max 10)
                Examples: "overtime", "grievance", "seniority", "health insurance"
            
            - key_provisions: array of specific rules or entitlements mentioned (max 5)
                Examples: ["30 days notice required", "4% wage increase", "8 paid holidays"]
            
            - job_titles_mentioned: array of specific job titles if mentioned (max 10)
                Examples: ["Network Administrator", "Staff Nurse II", "Electrician", "Foreman"]
                Use "None" if no specific job titles are mentioned
            
            - wage_amounts: array of specific dollar amounts or rates mentioned (max 5)
                Examples: ["$57.75/hour", "$2,000 bonus", "$25.08 minimum"]
                Use empty array if no specific amounts mentioned
            
            - percentage_increases: array of percentage changes mentioned (max 3)
                Examples: ["4% wage increase", "3.5% in year 2", "1.75% annual"]
                Use empty array if no percentages mentioned
            
            - contains_wage_data: boolean - does it include specific dollar amounts or wage scales?
            
            - contains_numerical_terms: boolean - does it mention specific numbers (days, hours, percentages)?

            2. contract: object with:
            - covered_employee_group: which employees does this apply to?
                Examples: "Faculty - Tenure System", "Nurses - RN", "Electricians", "Police Officers", "Clerical Staff", "Graduate Employees", "All Employees"
                Be as specific as possible
            
            - union_name: if mentioned, the union/local name
                Examples: "SEIU Local 73", "INA", "IBEW Local 134", "GEO Local 6297", "IFT-AFT AAUP Local 6456"
                Use "Not Specified" if not mentioned
            
            - campus_location: which campus does this apply to?
                Options: "Chicago", "Peoria", "Rockford", "All Campuses", "Not Specified"
                Look for location indicators in the text
            
            - effective_dates: object with contract period if mentioned
                Example: {{"start": "August 16, 2022", "end": "August 15, 2026"}}
                Use {{"start": "Not Specified", "end": "Not Specified"}} if dates not mentioned
            
            - exclusions: array of groups/titles explicitly excluded from coverage
                Examples: ["College of Medicine faculty", "Temporary employees", "Part-time under 25% FTE"]
                Use empty array if no exclusions mentioned
            
            - action_requirements: array of actions required by this provision (max 3)
                Examples: ["Submit written notice", "Attend pre-disciplinary meeting", "Provide 30 days notice"]
                Use ["No action required"] if this is informational only

            3. semantic: object with:
            - summary: concise 1-2 sentence plain language summary
                Focus on what employees/HR need to know
            
            - common_queries: 2-3 specific questions someone might ask about this content
                Examples:
                * "What's the hourly rate for electricians in 2025?"
                * "How many vacation days do nurses get?"
                * "What's the grievance procedure for faculty?"
                * "Which fees are waived for GEO members?"
            
            - related_topics: array of related subjects someone might also want to know about (max 3)
                Examples: ["Sick Leave", "Shift Differentials", "Union Dues", "Discipline Process"]

            
            Return ONLY valid JSON, nothing else.
            """
            
            # Call LLM with retries for rate limits
            for attempt in range(config.RETRY_LIMIT):
                try:
                    response = self.client.invoke(prompt)
                    content = response.content
                    
                    # Log the raw response
                    self.logger.debug(f"Raw LLM response: {content}")
                    
                    # Parse JSON safely
                    metadata = self._parse_json_safely(content)
                    return metadata
                    
                except Exception as e:
                    if "rate limit" in str(e).lower() and attempt < config.RETRY_LIMIT - 1:
                        self.logger.warning(f"Rate limit hit, retrying in {config.RETRY_DELAY} seconds")
                        time.sleep(config.RETRY_DELAY)
                    else:
                        self.logger.error(f"LLM error: {str(e)}")
                        self.logger.error(f"Last response content: {content if 'content' in locals() else 'N/A'}")
                        raise
            
            # Fallback if all retries fail
            return {

                "content": {
                    "content_type": {"primary": "Unknown", "subtypes": []},
                    "keywords": [],
                    "key_provisions": [],
                    "job_titles_mentioned": [],
                    "wage_amounts": [],
                    "percentage_increases": [],
                    "contains_wage_data": False,
                    "contains_numerical_terms": False
                },
                "contract": {
                    "covered_employee_group": "Unknown",
                    "union_name": "Not Specified",
                    "campus_location": "Not Specified",
                    "effective_dates": {"start": "Not Specified", "end": "Not Specified"},
                    "exclusions": [],
                    "action_requirements": []
                },
                "semantic": {
                    "summary": "Union labor agreement content",
                    "common_queries": [],
                    "related_topics": []
                }
            }
        except Exception as e:
            self.logger.error(f"Error generating combined metadata: {str(e)}")
            # Return default metadata on error
            return {

                "content": {
                    "content_type": {"primary": "Unknown", "subtypes": []},
                    "keywords": [],
                    "key_provisions": [],
                    "job_titles_mentioned": [],
                    "wage_amounts": [],
                    "percentage_increases": [],
                    "contains_wage_data": False,
                    "contains_numerical_terms": False
                },
                "contract": {
                    "covered_employee_group": "Unknown",
                    "union_name": "Not Specified",
                    "campus_location": "Not Specified",
                    "effective_dates": {"start": "Not Specified", "end": "Not Specified"},
                    "exclusions": [],
                    "action_requirements": []
                },
                "semantic": {
                    "summary": "Union labor agreement content",
                    "common_queries": [],
                    "related_topics": []
                }
            }


    def _generate_embedding_enhancement(self, text, content_metadata, contract_metadata, semantic_metadata):
        """Generate embedding enhancement fields."""
        try:
            # Create contextual prefix
            prefixes = []
            if "content_type" in content_metadata and "primary" in content_metadata["content_type"]:
                prefixes.append(f"[{content_metadata['content_type']['primary']}]")
            
            if "covered_employee_group" in contract_metadata:
                prefixes.append(f"[{contract_metadata['covered_employee_group']}]")
            
            # Collect keywords for TF-IDF enhancement
            keywords = []
            if "keywords" in content_metadata:
                keywords.extend(content_metadata["keywords"])
            
            if "job_titles_mentioned" in content_metadata:
                keywords.extend(content_metadata["job_titles_mentioned"])
            
            if "union_name" in contract_metadata and contract_metadata["union_name"] != "Not Specified":
                keywords.append(contract_metadata["union_name"])
            
            if "campus_location" in contract_metadata and contract_metadata["campus_location"] != "Not Specified":
                keywords.append(contract_metadata["campus_location"])
            
            # Remove duplicates and limit to 15 keywords
            unique_keywords = list(set(keywords))[:15]
            
            return {
                "contextual_prefix": " ".join(prefixes),
                "tf_idf_keywords": unique_keywords
            }
        except Exception as e:
            self.logger.error(f"Error generating embedding enhancement: {str(e)}")
            return {
                "contextual_prefix": "",
                "tf_idf_keywords": []
            }