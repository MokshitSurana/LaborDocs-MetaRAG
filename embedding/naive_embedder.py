import os
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

from embedding.base_embedder import BaseEmbedder

from gpu_utils import GPUVerifier

# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)

class NaiveEmbedder(BaseEmbedder):
    """Naive embedding strategy that only uses chunk content."""
    
    def _get_embedding_type_dir(self) -> str:
        """Get the directory name for naive embeddings."""
        return os.path.join(self.output_chunking_dir, "naive_embedding")
    
    def _get_embedding_type(self) -> str:
        """Get a string representing the embedding type."""
        return "naive"
    
    def _initialize_model(self):
        """Initialize the embedding model."""
        self.logger.info(f"Initializing SentenceTransformer model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            self.logger.info(f"Successfully loaded model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def _prepare_text_for_embedding(self, chunk: Dict[str, Any]) -> str:
        """Prepare text by combining chunk text with summary."""
        text = chunk.get("text", "")
        
        # Get summary from metadata
        summary = ""
        if "metadata" in chunk:
            if "semantic" in chunk["metadata"]:
                summary = chunk["metadata"]["semantic"].get("summary", "")
        
        # Combine: Summary first, then full text
        if summary:
            enhanced_text = f"{summary}\n\n{text}"
        else:
            enhanced_text = text
        
        return enhanced_text
    
    def _generate_embeddings(self, chunks: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
        """Generate embeddings for chunks using only the chunk content.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Tuple of (embeddings, chunk_ids, chunk_metadata)
        """
        self.logger.info(f"Generating naive embeddings for {len(chunks)} chunks")
        
        # Extract content and IDs
        texts = [self._prepare_text_for_embedding(chunk) for chunk in chunks]
        chunk_ids = [chunk.get("chunk_id", "") for chunk in chunks]
        
        # Prepare simplified metadata for storage
        chunk_metadata = []
        for chunk in chunks:
            # Create simplified metadata to save storage space
            metadata = {
                "chunk_id": chunk.get("chunk_id", ""),
                "text": chunk.get("text", ""),
                "document_id": chunk.get("document_id", ""),
                "document_name": chunk.get("document_name", "")
            }
            
            # Add only essential metadata fields if available
            if "metadata" in chunk:
                if "content" in chunk["metadata"]:
                    if "content_type" in chunk["metadata"]["content"]:
                        metadata["content_type"] = chunk["metadata"]["content"].get("content_type", {}).get("primary", "Unknown")

                if "contract" in chunk["metadata"]:
                    metadata["covered_employee_group"] = chunk["metadata"]["contract"].get("covered_employee_group", "Unknown")
                    metadata["union_name"] = chunk["metadata"]["contract"].get("union_name", "Not Specified")
                    metadata["campus_location"] = chunk["metadata"]["contract"].get("campus_location", "Not Specified")

                if "semantic" in chunk["metadata"]:
                    metadata["common_queries"] = chunk["metadata"]["semantic"].get("common_queries", [])
                    metadata["summary"] = chunk["metadata"]["semantic"].get("summary", "")
            
            chunk_metadata.append(metadata)
        
        # Generate embeddings
        try:
            self.logger.info("Generating embeddings with SentenceTransformer...")
            embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
            self.logger.info(f"Successfully generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
            
            return embeddings, chunk_ids, chunk_metadata
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise