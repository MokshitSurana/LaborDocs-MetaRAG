import os
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import re

from embedding.base_embedder import BaseEmbedder

from gpu_utils import GPUVerifier


# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)

class PrefixEmbedder(BaseEmbedder):
    """Prefix-injection embedding strategy that prepends metadata to content."""
    
    def _get_embedding_type_dir(self) -> str:
        """Get the directory name for prefix embeddings."""
        return os.path.join(self.output_chunking_dir, "prefix_fusion_embedding")
    
    def _get_embedding_type(self) -> str:
        """Get a string representing the embedding type."""
        return "prefix-fusion"
    
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
    
    def _initialize_model(self):
        """Initialize the embedding model."""
        self.logger.info(f"Initializing SentenceTransformer model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            self.logger.info(f"Successfully loaded model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

        
    def _format_service_context(self, metadata: Dict[str, Any]) -> str:
        """Extract and format contract context prefix (20% weight)."""
        employee_group = metadata.get("contract", {}).get("covered_employee_group", "General")
        return f"[Context:{employee_group}]"
        
    
    def _format_content_type(self, metadata: Dict[str, Any]) -> str:
        """Extract and format content type prefix (15% weight)."""
        content_type = metadata.get("content", {}).get("content_type", {}).get("primary", "General")
        return f"[{content_type}]"
    
    def _format_employee_group(self, metadata: Dict[str, Any]) -> str:
        """Extract and format employee group prefix (10% weight)."""
        employee_group = metadata.get("contract", {}).get("covered_employee_group", "General")
        # Remove spaces for tokenization efficiency
        employee_group = employee_group.replace(" ", "").replace("-", "")
        return f"[{employee_group}]"
    
    
    def _format_wage_data_presence(self, metadata: Dict[str, Any]) -> str:
        """Extract and format wage data presence prefix (10% weight)."""
        has_wage_data = metadata.get("content", {}).get("contains_wage_data", False)
        if not has_wage_data:
            return "[Wages:None]"
        
        return f"[Wages:Present]"
    
    
    def _format_common_query(self, metadata: Dict[str, Any]) -> str:
        """Extract and format common query prefix (20% weight)."""
        queries = metadata.get("semantic", {}).get("common_queries", [])
        if not queries:
            return ""
        
        # Use first query
        query = queries[0]
        
        # Normalize: remove spaces, question marks, convert to CamelCase
        q_text = query.replace("?", "").strip()
        
        # Convert to CamelCase
        q_text = re.sub(r'[^a-zA-Z0-9]', ' ', q_text)  # Replace non-alphanumeric with space
        words = q_text.split()
        if words:
            q_text = words[0].lower() + ''.join(w.capitalize() for w in words[1:])
        
        # Truncate if too long
        if len(q_text) > 50:
            q_text = q_text[:50]
            
        return f"[Q:{q_text}]"
    

    def _format_prefixes(self, metadata: Dict[str, Any]) -> str:
        """Format metadata as prefixes for embedding."""
        # Contract context prefix (20%) - union/campus
        context_prefix = self._format_service_context(metadata)
        
        # Content type prefix (15%)
        content_type_prefix = self._format_content_type(metadata)
        
        # Employee group prefix (10%)
        employee_group_prefix = self._format_employee_group(metadata)
        
        # Wage data presence prefix (10%)
        wage_prefix = self._format_wage_data_presence(metadata)
        
        # Common query prefix (20%)
        query_prefix = self._format_common_query(metadata)
        
        # Combine all prefixes
        all_prefixes = [context_prefix, content_type_prefix, 
                    employee_group_prefix, wage_prefix, query_prefix]
        
        return " ".join([p for p in all_prefixes if p])

    
    def _generate_embeddings(self, chunks: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
        """Generate embeddings for chunks using prefix injection.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Tuple of (embeddings, chunk_ids, chunk_metadata)
        """
        self.logger.info(f"Generating prefix-injected embeddings for {len(chunks)} chunks")
        
        # Extract content, IDs, and prepare augmented texts
        augmented_texts = []
        chunk_ids = []
        
        for chunk in chunks:
            # Format prefixes from metadata
            prefixes = self._format_prefixes(chunk.get("metadata", {}))
            
            # Combine prefixes with chunk content
            text = self._prepare_text_for_embedding(chunk)
            augmented_text = f"{prefixes} {text}"
            
            augmented_texts.append(augmented_text)
            chunk_ids.append(chunk.get("chunk_id", ""))
        
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
            embeddings = self.model.encode(augmented_texts, show_progress_bar=True, batch_size=32)
            self.logger.info(f"Successfully generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
            
            return embeddings, chunk_ids, chunk_metadata
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise