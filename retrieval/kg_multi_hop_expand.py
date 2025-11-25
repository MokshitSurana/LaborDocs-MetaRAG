#!/usr/bin/env python3
"""
Knowledge Graph Multi-Hop Query Expansion
"""

import re
from typing import List
from neo4j import GraphDatabase

# Your Neo4j credentials
NEO4J_URI = "neo4j+s://7a4fcdb3.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PWD = "PTT11bEFv3EtN91R4ReV_2Tc55gcgpqp-dcFJtn4oAk"

_driver = None


def get_neo4j_driver():
    """Reusable Neo4j driver with safe caching."""
    global _driver
    if _driver is not None:
        return _driver

    try:
        _driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PWD),
            connection_timeout=10
        )
        return _driver
    except Exception as e:
        print(f"[Neo4j ERROR] Could not connect → {e}")
        return None


def extract_keywords(text: str) -> List[str]:
    """Extract meaningful keywords from the query."""
    stop_words = {
        'what', 'is', 'the', 'for', 'of', 'in', 'are', 'to', 'a', 'an', 
        'salary', 'wage', 'how', 'much', 'must', 'can', 'should', 'will',
        'be', 'have', 'has', 'do', 'does', 'covered', 'under', 'by', 'at'
    }
    
    # Clean text
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    
    # Filter stop words and short tokens
    keywords = [w for w in tokens if w not in stop_words and len(w) > 2]
    
    return keywords


def kg_multi_hop_expand(query: str, max_nodes: int = 8) -> str:
    """
    Multi-hop query expansion using Knowledge Graph.
    
    1. Extracts keywords from query
    2. Finds matching Entity nodes in Neo4j
    3. Traverses 1-2 hops to find related context
    4. Returns discovered terms to augment search
    
    Args:
        query: User's question
        max_nodes: Maximum number of related terms to return
        
    Returns:
        String of space-separated related terms found in KG
    """
    driver = get_neo4j_driver()
    if driver is None:
        print("[KG] Neo4j driver not available")
        return ""

    # Extract keywords
    keywords = extract_keywords(query)
    if not keywords:
        print("[KG] No keywords extracted from query")
        return ""
    
    print(f"[KG] Query keywords: {keywords}")

    # Cypher query for multi-hop expansion
    cypher = """
    UNWIND $keywords AS kw
    
    // Step 1: Find anchor nodes matching keywords
    MATCH (anchor:Entity)
    WHERE toLower(anchor.name) CONTAINS kw
    
    // Step 2: Traverse 1-2 hops to related entities
    // Use your RELATED relationship
    MATCH path = (anchor)-[:RELATED*1..2]-(related:Entity)
    
    // Step 3: Filter out generic/irrelevant types
    WHERE related.type IN ['position', 'wage', 'union', 'date', 'step']
      AND related.name IS NOT NULL
    
    // Step 4: Return distinct related terms
    RETURN DISTINCT related.name as term, related.type as type
    LIMIT $max_nodes
    """

    try:
        with driver.session() as session:
            result = session.run(
                cypher, 
                keywords=keywords, 
                max_nodes=max_nodes
            )
            
            records = list(result)
            
            if not records:
                print(f"[KG] No related entities found for: {keywords}")
                return ""
            
            # Extract terms
            terms = [record["term"] for record in records if record["term"]]
            
            # Clean and deduplicate
            cleaned = list(set([str(t).strip() for t in terms if t]))
            
            print(f"[KG] Found {len(cleaned)} related terms: {cleaned[:5]}...")
            
            # Return as space-separated string
            return " ".join(cleaned)

    except Exception as e:
        print(f"[KG ERROR] Multi-hop expansion failed: {e}")
        return ""


def kg_filter_by_contract(query: str) -> List[str]:
    """
    Use KG to identify which contract(s) are relevant to the query.
    
    Returns list of contract names (e.g., ["GEO-CBA 2022-2025.txt"])
    """
    driver = get_neo4j_driver()
    if driver is None:
        return []

    keywords = extract_keywords(query)
    if not keywords:
        return []
    
    # Query to find contracts related to query keywords
    cypher = """
    UNWIND $keywords AS kw
    
    // Find entities matching keywords
    MATCH (e:Entity)
    WHERE toLower(e.name) CONTAINS kw
    
    // Find which contracts these entities belong to
    MATCH (c:Entity {type: 'contract'})-[:RELATED*1..2]-(e)
    
    // Return contract names
    RETURN DISTINCT c.name as contract_name
    LIMIT 3
    """
    
    try:
        with driver.session() as session:
            result = session.run(cypher, keywords=keywords)
            contracts = [rec["contract_name"] for rec in result if rec["contract_name"]]
            
            if contracts:
                print(f"[KG] Identified relevant contracts: {contracts}")
            
            return contracts
    
    except Exception as e:
        print(f"[KG ERROR] Contract filtering failed: {e}")
        return []