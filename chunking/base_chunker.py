from abc import ABC, abstractmethod
import json
import os
import re  # Added re for regex cleaning
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple

from utils.logger import setup_logger
from gpu_utils import GPUVerifier

# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)

class BaseChunker(ABC):
    """Abstract base class for text chunking strategies."""
    
    def __init__(
        self,
        min_chunk_size: int = 50,
        max_chunk_size: int = 500,
        overlap: int = 50,
        output_dir: str = "chunk_output"
    ):
        """Initialize the base chunker."""
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.output_dir = output_dir
        self.logger = setup_logger(self.__class__.__name__)
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            self.logger.info(f"Created output directory: {output_dir}")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing excessive formatting, Table of Contents artifacts, 
        and repeated delimiters.
        """
        # 1. Remove Table of Contents leader dots (e.g., "Purpose .................. 1")
        # Matches 3+ dots, optional whitespace, and optional page number at the end
        text = re.sub(r'\.{3,}\s*\d*', ' ', text)
        
        # 2. Remove common headers that clutter chunks (Customize as needed)
        text = re.sub(r'(?i)Table of Contents', '', text)
        text = re.sub(r'PAGE\s*$', '', text, flags=re.MULTILINE)
        
        # 3. Collapse multiple newlines into max 2 (preserves paragraph structure but removes gaps)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 4. Collapse multiple spaces/tabs into one single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # 5. Remove underscores often used for signature lines (e.g. "__________")
        text = re.sub(r'_{3,}', ' ', text)

        return text.strip()

    def process_document(self, document_path: str) -> Dict[str, Any]:
        """Process a document and create chunks."""
        self.logger.info(f"Processing document: {document_path}")
        
        try:
            # Generate a unique document ID
            document_id = str(uuid.uuid4())
            document_name = os.path.basename(document_path)
            
            # Read document content
            with open(document_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # === APPLY CLEANING HERE ===
            self.logger.info("Cleaning text artifacts...")
            text = self._clean_text(text)
            
            # Create chunks
            self.logger.info("Creating chunks...")
            chunks, document_metadata = self._create_chunks(text, document_id)
            
            # Remove invalid chunks
            valid_chunks = self._filter_valid_chunks(chunks)
            self.logger.info(f"Created {len(valid_chunks)} valid chunks out of {len(chunks)} total chunks")
            
            # Create result document
            result = {
                "document_id": document_id,
                "document_name": document_name,
                "chunks": valid_chunks,
                "metadata": {
                    "total_chunks": len(valid_chunks),
                    "avg_chunk_size_words": sum(len(chunk["text"].split()) for chunk in valid_chunks) / max(1, len(valid_chunks)),
                    "chunking_method": self.__class__.__name__,
                    "processed_at": datetime.now().isoformat(),
                    **document_metadata
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing document {document_path}: {str(e)}")
            raise

    def save_chunks(self, result: Dict[str, Any], output_path: str = None) -> str:
        """Save chunks to a JSON file (preventing duplicate suffixes)."""
        try:
            document_id = result["document_id"]
            document_name = result["document_name"]
            
            # Create a more descriptive filename only if output_path not provided
            if output_path is None or not output_path.endswith(".json"):
                base_name = os.path.splitext(document_name)[0]
                method_name = self.__class__.__name__.lower().replace('chunker', '')
                
                # Add specific parameters based on chunker type
                params = ""
                if method_name == "semantic":
                    params = f"_p{getattr(self, 'percentile_threshold', 95)}"
                elif method_name == "recursive":
                    params = f"_{getattr(self, 'split_method', 'length')}"
                
                filename = f"{base_name}_{method_name}{params}_chunks.json"
                output_path = os.path.join(self.output_dir, filename)
            
            # Write the file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            
            self.logger.info(f"Saved chunks to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error saving chunks: {str(e)}")
            raise

    def _filter_valid_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out invalid chunks (null or incoherent)."""
        valid_chunks = []
        for chunk in chunks:
            # 1. Skip empty chunks
            if not chunk.get("text") or len(chunk["text"].strip()) == 0:
                continue
            
            # 2. Calculate Word Count
            words = chunk["text"].split()
            num_words = len(words)
            
            # 3. Validate Size
            if num_words < self.min_chunk_size:
                # self.logger.debug(f"Skipping too small chunk: {chunk.get('chunk_id')} ({num_words} words)")
                continue
            
            valid_chunks.append(chunk)
            
        return valid_chunks
    
    @abstractmethod
    def _create_chunks(self, text: str, document_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Abstract method to create chunks from text."""
        pass