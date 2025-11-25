#!/usr/bin/env python3
"""
Hybrid MetaRAG Retriever
Implements parallel RAG + KG retrieval with Reciprocal Rank Fusion (RRF)

Architecture:
1. RAG Path: Semantic search using embeddings
2. KG Path: Multi-hop graph traversal
3. RRF Fusion: Merge and boost overlapping results
4. Context Enrichment: Add neighbor chunks
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict
import faiss
import pickle
import json
import os

from retrieval.base_retriever import BaseRetriever
from utils.logger import setup_logger
from gpu_utils import GPUVerifier

# Import KG functions
from retrieval.kg_multi_hop_expand import (
    kg_multi_hop_expand, 
    extract_keywords,
    get_neo4j_driver
)

gpu_verifier = GPUVerifier(require_gpu=True)


class HybridMetaRAGRetriever(BaseRetriever):
    """
    Hybrid retriever combining RAG (semantic search) and KG (graph traversal)
    with Reciprocal Rank Fusion for result merging.
    """
    
    def __init__(
        self,
        embedding_dir: str = "embeddings_output",
        chunking_type: str = "semantic",
        embedding_type: str = "naive_embedding",
        model_name: str = "Snowflake/arctic-embed-s",
        top_k: int = 8,
        rag_k: int = 8,
        kg_k: int = 8,
        rrf_k: int = 60,
        rag_weight: float = 1.0,  # <--- NEW: High trust in RAG
        kg_weight: float = 0.2, # <--- NEW: Lower trust in KG
        enable_context_enrichment: bool = True
    ):
        """
        Initialize Hybrid MetaRAG retriever.
        
        Args:
            embedding_dir: Base directory containing embeddings
            chunking_type: Type of chunking (semantic, naive, recursive)
            embedding_type: Type of embedding to use for RAG path:
                - 'naive_embedding': ContentRetriever (just content)
                - 'prefix_fusion_embedding': PrefixRetriever (metadata prefixes)
                - 'tfidf_embedding': TfidfRetriever (TF-IDF weighted)
            model_name: Name of embedding model
            top_k: Final number of results after fusion
            rag_k: Number of results from RAG path
            kg_k: Number of results from KG path
            rrf_k: RRF constant (typically 60)
            enable_context_enrichment: Whether to add neighbor chunks
        """
        super().__init__(
            embedding_dir=embedding_dir,
            chunking_type=chunking_type,
            embedding_type=embedding_type,
            model_name=model_name,
            top_k=top_k
        )
        
        self.rag_k = rag_k
        self.kg_k = kg_k
        self.rrf_k = rrf_k
        self.enable_context_enrichment = enable_context_enrichment
        self.rag_weight = rag_weight
        self.kg_weight = kg_weight
        
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info(f"Hybrid MetaRAG initialized:")
        self.logger.info(f"  - Embedding Type: {embedding_type}")
        self.logger.info(f"  - Chunking Type: {chunking_type}")
        self.logger.info(f"  - RAG_K={rag_k}, KG_K={kg_k}, RRF_K={rrf_k}")
        self.logger.info(f"  - Context Enrichment: {enable_context_enrichment}")
    
    def _load_index(self):
        """Load FAISS index (inherited from BaseRetriever but needs explicit call)."""
        if self.index is not None:
            return
        
        index_path = os.path.join(self.embedding_type_dir, "index.faiss")
        mapping_path = os.path.join(self.embedding_type_dir, "id_mapping.pkl")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        
        self.index = faiss.read_index(index_path)
        self.logger.info(f"Loaded FAISS index from {index_path}")
        
        with open(mapping_path, "rb") as f:
            mappings = pickle.load(f)
            self.id_to_index = mappings["id_to_index"]
            self.index_to_id = mappings["index_to_id"]
        
        self.logger.info(f"Loaded ID mappings from {mapping_path}")
        self.logger.info(f"FAISS index contains {self.index.ntotal} vectors")
        self.logger.info(f"FAISS index dimension: {self.index.d}")
    
    def _load_metadata(self):
        """Load chunk metadata."""
        if self.id_to_metadata:
            return
        
        metadata_path = os.path.join(self.embedding_type_dir, "metadata.json")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.id_to_metadata = json.load(f)
        
        self.logger.info(f"Loaded metadata from {metadata_path}")
    
    def _prepare_query(self, query: str) -> np.ndarray:
        """
        Prepare query embedding (required by BaseRetriever).
        For HybridMetaRAG, this is used by the RAG path.
        
        Args:
            query: Query string
            
        Returns:
            Query embedding vector
        """
        self._load_embedding_model()
        query_embedding = self.embedding_model.encode([query])
        return self.normalize_vector(query_embedding)
    
    def _rag_retrieve(self, query: str) -> List[Tuple[str, float, int]]:
        """
        RAG path: Semantic search using embeddings.
        
        Returns:
            List of (chunk_id, score, rank) tuples
        """
        self.logger.info(f"[RAG PATH] Retrieving {self.rag_k} results via semantic search...")
        
        # Use _prepare_query to get embedding
        query_vector = self._prepare_query(query)
        
        # Search FAISS index
        scores, indices = self.index.search(query_vector, self.rag_k)
        
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx == -1:
                continue
            
            chunk_id = self.index_to_id.get(idx)
            if chunk_id:
                results.append((chunk_id, float(score), rank))
        
        self.logger.info(f"[RAG PATH] Retrieved {len(results)} results")
        return results
    
    def _kg_retrieve(self, query: str) -> List[Tuple[str, float, int]]:
        """
        KG path: Multi-hop graph traversal + semantic search.
        
        Returns:
            List of (chunk_id, score, rank) tuples
        """
        self.logger.info(f"[KG PATH] Retrieving {self.kg_k} results via graph traversal...")
        
        # Step 1: Multi-hop expansion to get related terms
        kg_terms = kg_multi_hop_expand(query, max_nodes=10)
        
        if not kg_terms:
            self.logger.warning("[KG PATH] No KG expansion terms found, using original query")
            kg_enhanced_query = query
        else:
            self.logger.info(f"[KG PATH] KG expansion found: {kg_terms[:100]}...")
            kg_enhanced_query = f"{query} {kg_terms}"
        
        # Step 2: Semantic search with KG-enhanced query using _prepare_query
        self._load_embedding_model()
        kg_query_embedding = self.embedding_model.encode([kg_enhanced_query])
        kg_query_vector = self.normalize_vector(kg_query_embedding)
        
        scores, indices = self.index.search(kg_query_vector, self.kg_k)
        
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx == -1:
                continue
            
            chunk_id = self.index_to_id.get(idx)
            if chunk_id:
                results.append((chunk_id, float(score), rank))
        
        self.logger.info(f"[KG PATH] Retrieved {len(results)} results")
        return results
    
    def _reciprocal_rank_fusion(
        self, 
        rag_results: List[Tuple[str, float, int]], 
        kg_results: List[Tuple[str, float, int]]
    ) -> List[str]:
        """
        Reciprocal Rank Fusion (RRF) to merge RAG and KG results.
        
        Formula: score = 1 / (k + rank)
        Documents appearing in BOTH lists get BOOSTED scores.
        
        Args:
            rag_results: Results from RAG path
            kg_results: Results from KG path
            
        Returns:
            List of chunk_ids sorted by fused score
        """
        self.logger.info("[RRF] Fusing RAG and KG results...")
        
        # Calculate RRF scores
        rrf_scores = defaultdict(float)
        
        # Add RAG scores
        for chunk_id, score, rank in rag_results:
            score = 1.0 / (self.rrf_k + rank)
            rrf_scores[chunk_id] += (score * self.rag_weight)
        
        # Add KG scores (documents in BOTH get higher combined scores)
        for chunk_id, score, rank in kg_results:
            score = 1.0 / (self.rrf_k + rank)
            rrf_scores[chunk_id] += (score * self.kg_weight)
        
        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Count overlapping documents
        rag_ids = {chunk_id for chunk_id, _, _ in rag_results}
        kg_ids = {chunk_id for chunk_id, _, _ in kg_results}
        overlap = rag_ids & kg_ids
        
        self.logger.info(
            f"[RRF] Fused {len(sorted_results)} unique documents "
            f"({len(overlap)} appeared in BOTH paths - BOOSTED!)"
        )
        
        # Return top_k chunk IDs
        return [chunk_id for chunk_id, score in sorted_results[:self.top_k]]
    
    def _enrich_with_context(self, chunk_ids: List[str]) -> List[str]:
        """
        Context enrichment: Add neighbor chunks (±1) from same document.
        
        Args:
            chunk_ids: List of chunk IDs
            
        Returns:
            Enriched list with neighbor chunks added
        """
        if not self.enable_context_enrichment:
            return chunk_ids
        
        self.logger.info("[CONTEXT] Enriching with neighbor chunks...")
        
        enriched_ids = set(chunk_ids)  # Start with original chunks
        
        for chunk_id in chunk_ids:
            # Parse chunk_id: "document_id_chunk_N"
            try:
                parts = chunk_id.rsplit('_chunk_', 1)
                if len(parts) != 2:
                    continue
                
                doc_id = parts[0]
                chunk_num = int(parts[1])
                
                # Add previous chunk
                if chunk_num > 0:
                    prev_id = f"{doc_id}_chunk_{chunk_num - 1}"
                    if prev_id in self.id_to_index:
                        enriched_ids.add(prev_id)
                
                # Add next chunk
                next_id = f"{doc_id}_chunk_{chunk_num + 1}"
                if next_id in self.id_to_index:
                    enriched_ids.add(next_id)
            
            except (ValueError, IndexError):
                continue
        
        added = len(enriched_ids) - len(chunk_ids)
        self.logger.info(f"[CONTEXT] Added {added} neighbor chunks (total: {len(enriched_ids)})")
        
        return list(enriched_ids)
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Main retrieval method using Hybrid MetaRAG.
        
        Workflow:
        1. Parallel RAG + KG retrieval
        2. RRF fusion
        3. Context enrichment
        4. Load metadata and return
        
        Args:
            query: User query string
            
        Returns:
            List of retrieved chunks with metadata
        """
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"HYBRID METARAG RETRIEVAL")
        self.logger.info(f"Query: {query}")
        self.logger.info(f"{'='*70}")
        
        # Load index and metadata
        self._load_index()
        self._load_metadata()
        
        # Step 1: Parallel retrieval
        self.logger.info("\n[STEP 1] PARALLEL RETRIEVAL")
        rag_results = self._rag_retrieve(query)
        kg_results = self._kg_retrieve(query)
        
        # Step 2: RRF Fusion
        self.logger.info("\n[STEP 2] RECIPROCAL RANK FUSION")
        fused_chunk_ids = self._reciprocal_rank_fusion(rag_results, kg_results)
        
        # Step 3: Context Enrichment
        self.logger.info("\n[STEP 3] CONTEXT ENRICHMENT")
        enriched_chunk_ids = self._enrich_with_context(fused_chunk_ids)
        
        # Step 4: Load full metadata
        self.logger.info("\n[STEP 4] LOADING METADATA")
        results = []
        for chunk_id in enriched_chunk_ids[:self.top_k]:  # Respect top_k
            metadata = self.id_to_metadata.get(chunk_id, {})
            
            if not metadata:
                continue
            
            # Filter out "unknown" sources
            doc_name = metadata.get("document_name", "")
            if doc_name.lower() == "unknown" or not doc_name:
                continue
            
            results.append({
                "chunk_id": chunk_id,
                "text": metadata.get("text", ""),
                "document_name": doc_name,
                "document_id": metadata.get("document_id", ""),
                "score": 1.0,  # Placeholder (RRF score if needed)
                "retrieval_method": "hybrid_metarag",
                "content_type": metadata.get("content_type", "Unknown"),
                "covered_employee_group": metadata.get("covered_employee_group", "Unknown"),
                "union_name": metadata.get("union_name", "Not Specified"),
                "summary": metadata.get("summary", ""),
                "common_queries": metadata.get("common_queries", [])
            })
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"[+] RETRIEVAL COMPLETE: {len(results)} chunks")
        self.logger.info(f"{'='*70}\n")
        
        return results