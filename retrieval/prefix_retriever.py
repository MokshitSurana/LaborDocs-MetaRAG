import numpy as np
import re
from typing import List, Dict, Any

from retrieval.base_retriever import BaseRetriever
from gpu_utils import GPUVerifier
# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)


class PrefixFusionRetriever(BaseRetriever):
    """Retriever that uses prefix-fusion embeddings."""
    
    def __init__(
        self,
        embedding_dir: str = "embeddings_output",
        chunking_type: str = "semantic",
        model_name: str = "Snowflake/arctic-embed-s",
        top_k: int = 10
    ):
        """Initialize the prefix-fusion retriever.
        
        Args:
            embedding_dir: Base directory containing embeddings
            chunking_type: Type of chunking (semantic, naive, recursive)
            model_name: Name of the embedding model to use
            top_k: Number of results to retrieve
        """
        # Use prefix-fusion embeddings
        super().__init__(
            embedding_dir=embedding_dir,
            chunking_type=chunking_type,
            embedding_type="prefix_fusion_embedding",
            model_name=model_name,
            top_k=top_k
        )
    
        
    def _format_content_type_prefix_from_query(self, query: str) -> str:
        """Extract and format content type prefix from query."""
        query_lower = query.lower()
        
        # Detect content type based on query keywords
        if any(q in query_lower for q in ["wage", "salary", "pay", "hourly", "rate", "compensation"]):
            return "[Type:WageTable]"
        elif any(q in query_lower for q in ["benefit", "insurance", "health", "vacation", "leave", "holiday"]):
            return "[Type:Benefits]"
        elif any(q in query_lower for q in ["hour", "shift", "schedule", "overtime", "weekend"]):
            return "[Type:WorkingConditions]"
        elif any(q in query_lower for q in ["grievance", "arbitration", "dispute", "complaint"]):
            return "[Type:DisputeResolution]"
        elif any(q in query_lower for q in ["hire", "promotion", "probation", "appointment", "termination"]):
            return "[Type:EmploymentTerms]"
        elif any(q in query_lower for q in ["right", "protection", "discrimination", "union"]):
            return "[Type:RightsandProtections]"
        else:
            return "[Type:ContractProvisions]"
        
    
    def _format_contract_context_prefix(self, query: str) -> str:
        """Extract and format contract context prefix from query."""
        query_lower = query.lower()
        
        # Look for employee groups
        employee_groups = {
            "faculty": "Faculty",
            "tenure": "Faculty-TenureSystem",
            "non-tenure": "Faculty-NonTenure",
            "nurse": "Nurses",
            "rn": "Nurses-RN",
            "lpn": "Nurses-LPN",
            "electrician": "Electricians",
            "carpenter": "Carpenters",
            "plumber": "Plumbers",
            "pipefitter": "Pipefitters",
            "police": "PoliceOfficers",
            "security": "SecurityPersonnel",
            "clerical": "ClericalStaff",
            "technical": "TechnicalStaff",
            "service": "ServiceEmployees",
            "graduate": "GraduateEmployees",
            "geo": "GraduateEmployees"
        }
        
        # Check for employee group mentions
        for keyword, group in employee_groups.items():
            if keyword in query_lower:
                return f"[Context:{group}]"
        
        # Check for union mentions
        unions = {
            "seiu": "SEIULocal73",
            "ina": "INA",
            "ibew": "IBEW",
            "iuoe": "IUOE",
            "ift": "IFT-AFT"
        }
        
        for keyword, union in unions.items():
            if keyword in query_lower:
                return f"[Context:{union}]"
        
        return ""

    
    def _format_campus_prefix(self, query: str) -> str:
        """Extract and format campus location prefix from query."""
        query_lower = query.lower()
        
        if "peoria" in query_lower:
            return "[Campus:Peoria]"
        elif "rockford" in query_lower:
            return "[Campus:Rockford]"
        elif "dscc" in query_lower:
            return "[Campus:DSCC]"
        elif "chicago" in query_lower or "uic" in query_lower:
            return "[Campus:Chicago]"
        
        return ""

    
    def _format_query_prefixes(self, query: str) -> str:
        """Format query with prefixes for embedding."""
        # Content type prefix (based on query intent)
        content_type_prefix = self._format_content_type_prefix_from_query(query)
        
        # Contract context prefix (employee group or union)
        context_prefix = self._format_contract_context_prefix(query)
        
        # Campus location prefix
        campus_prefix = self._format_campus_prefix(query)
        
        # Question prefix (keep same logic)
        q_text = query.replace("?", "").strip()
        q_text = re.sub(r'[^a-zA-Z0-9]', ' ', q_text)
        words = q_text.split()
        if words:
            q_text = words[0].lower() + ''.join(w.capitalize() for w in words[1:])
        
        if len(q_text) > 50:
            q_text = q_text[:50]
            
        question_prefix = f"[Q:{q_text}]"
        
        # Combine all prefixes
        all_prefixes = [content_type_prefix, context_prefix, campus_prefix, question_prefix]
        
        return " ".join([p for p in all_prefixes if p])

    
    def _prepare_query(self, query: str) -> np.ndarray:
        """Prepare a query for searching using prefix-fusion embedding.
        
        Args:
            query: The query string
            
        Returns:
            A query vector
        """
        # Load embedding model if needed
        self._load_embedding_model()
        
        # Format prefixes
        prefixed_query = self._format_query_prefixes(query) + " " + query
        
        # Embed query
        query_vector = self.embedding_model.encode([prefixed_query])
        
        # Normalize
        query_vector = self.normalize_vector(query_vector)
        
        return query_vector