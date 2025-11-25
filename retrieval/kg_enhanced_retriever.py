#!/usr/bin/env python3
"""
Hybrid KG-Enhanced Retriever
Combines semantic search with knowledge graph multi-hop expansion.
"""

import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

from retrieval.base_retriever import BaseRetriever
from utils.logger import setup_logger
from gpu_utils import GPUVerifier

# Import your KG functions
from kg_multi_hop_expand import kg_multi_hop_expand

gpu_verifier = GPUVerifier(require_gpu=True)


class KGEnhancedRetriever(BaseRetriever):
    """
    Retriever that uses Knowledge Graph multi-hop expansion
    to enhance semantic search results.
    """
    
    def __init__(
        self,
        embedding_dir: str = "embeddings_output",
        chunking_type: str = "semantic",
        embedding_type: str = "naive_embedding",
        model_name: str = "Snowflake/arctic-embed-s",
        top_k: int = 5,
        kg_expansion_nodes: int = 8,
        kg_weight: float = 0.3
    ):
        """
        Initialize KG-enhanced retriever.
        
        Args:
            embedding_dir: Base directory containing embeddings
            chunking_type: Type of chunking (semantic, naive, recursive)
            embedding_type: Type of embedding to use
            model_name: Name of the embedding model
            top_k: Number of results to retrieve
            kg_expansion_nodes: Max nodes to expand in KG
            kg_weight: Weight for KG-enhanced query (0.0-1.0)
        """
        super().__init__(
            embedding_dir=embedding_dir,
            chunking_type=chunking_type,
            embedding_type=embedding_type,
            model_name=model_name,
            top_k=top_k
        )
        
        self.kg_expansion_nodes = kg_expansion_nodes
        self.kg_weight = kg_weight
        self.logger = setup_logger(self.__class__.__name__)
    
    def _prepare_query(self, query: str) -> np.ndarray:
        """
        Prepare query using KG expansion + semantic embedding.
        
        Args:
            query: The query string
            
        Returns:
            Enhanced query vector
        """
        # Load embedding model if needed
        self._load_embedding_model()
        
        # Step 1: Get KG expansion terms
        self.logger.info(f"Expanding query with KG (max_nodes={self.kg_expansion_nodes})")
        kg_terms = kg_multi_hop_expand(query, max_nodes=self.kg_expansion_nodes)
        
        if kg_terms:
            self.logger.info(f"KG found related terms: {kg_terms[:100]}...")
            
            # Step 2: Create enhanced query
            enhanced_query = f"{query} {kg_terms}"
        else:
            self.logger.warning("KG expansion returned no terms, using original query")
            enhanced_query = query
        
        # Step 3: Generate embeddings for both
        original_embedding = self.embedding_model.encode([query])
        enhanced_embedding = self.embedding_model.encode([enhanced_query])
        
        # Step 4: Blend embeddings
        # Weight: (1-kg_weight) * original + kg_weight * enhanced
        blended_embedding = (
            (1 - self.kg_weight) * self.normalize_vector(original_embedding) +
            self.kg_weight * self.normalize_vector(enhanced_embedding)
        )
        
        # Final normalization
        return self.normalize_vector(blended_embedding)
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve with KG enhancement and add KG context to results.
        
        Args:
            query: The query string
            
        Returns:
            List of retrieved chunks with KG metadata
        """
        # Get KG terms for metadata
        kg_terms = kg_multi_hop_expand(query, max_nodes=self.kg_expansion_nodes)
        
        # Perform retrieval
        results = super().retrieve(query)
        
        # Add KG metadata to results
        for result in results:
            result["kg_enhanced"] = bool(kg_terms)
            result["kg_expansion_terms"] = kg_terms[:200] if kg_terms else ""
        
        self.logger.info(f"Retrieved {len(results)} results with KG enhancement")
        
        return results