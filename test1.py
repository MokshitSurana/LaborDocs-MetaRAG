#!/usr/bin/env python3
"""
Debug script to check enriched chunks and embedder flow.
"""
import json
import glob
import os

print("=" * 80)
print("DEBUGGING ENRICHED CHUNKS AND EMBEDDER FLOW")
print("=" * 80)

# Step 1: Check if enriched chunks directory exists
input_dir = "metadata_gen_output/semantic_chunks_metadata"
print(f"\n📁 Checking directory: {input_dir}")
print(f"   Exists: {os.path.exists(input_dir)}")

if not os.path.exists(input_dir):
    print("❌ ERROR: Directory does not exist!")
    exit(1)

# Step 2: Find enriched chunk files
json_files = glob.glob(os.path.join(input_dir, "*_enriched_chunks.json"))
print(f"\n📄 Found {len(json_files)} enriched chunk files:")
for f in json_files:
    print(f"   - {os.path.basename(f)}")

if not json_files:
    print("❌ ERROR: No enriched chunk files found!")
    exit(1)

# Step 3: Check first file structure
first_file = json_files[0]
print(f"\n🔍 Examining: {os.path.basename(first_file)}")

try:
    with open(first_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n✅ File loaded successfully")
    print(f"   Type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"   Root keys: {list(data.keys())}")
        print(f"   document_id: {data.get('document_id', 'MISSING')}")
        print(f"   document_name: {data.get('document_name', 'MISSING')}")
        
        if "chunks" in data:
            chunks = data["chunks"]
            print(f"   Number of chunks: {len(chunks)}")
            
            if chunks:
                first_chunk = chunks[0]
                print(f"\n   First chunk:")
                print(f"      chunk_id: {first_chunk.get('chunk_id', 'MISSING')}")
                print(f"      Has 'text': {'text' in first_chunk}")
                print(f"      Has 'metadata': {'metadata' in first_chunk}")
                print(f"      Has 'document_name' (before injection): {'document_name' in first_chunk}")
        else:
            print("   ❌ No 'chunks' key found!")
    else:
        print(f"   ❌ Unexpected data type: {type(data)}")

except Exception as e:
    print(f"❌ ERROR loading file: {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 4: Simulate what base_embedder does
print(f"\n🔧 Simulating base_embedder injection...")

document_id = data.get("document_id", "")
document_name = data.get("document_name", "")
chunks = data.get("chunks", [])

print(f"   Extracted document_name: '{document_name}'")
print(f"   Extracted document_id: '{document_id}'")

# Inject into chunks
for chunk in chunks:
    chunk["document_id"] = document_id
    chunk["document_name"] = document_name

# Check first chunk after injection
if chunks:
    first_chunk = chunks[0]
    print(f"\n   After injection - First chunk:")
    print(f"      document_name: '{first_chunk.get('document_name', 'MISSING')}'")
    print(f"      document_id: '{first_chunk.get('document_id', 'MISSING')}'")

# Step 5: Simulate what tfidf_embedder does
print(f"\n🔧 Simulating tfidf_embedder metadata creation...")

if chunks:
    chunk = chunks[0]
    metadata = {
        "chunk_id": chunk.get("chunk_id", ""),
        "text": chunk.get("text", "")[:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get("text", ""),
        "document_id": chunk.get("document_id", ""),
        "document_name": chunk.get("document_name", "")
    }
    
    print(f"   Created metadata:")
    print(f"      chunk_id: '{metadata['chunk_id']}'")
    print(f"      document_id: '{metadata['document_id']}'")
    print(f"      document_name: '{metadata['document_name']}'")
    print(f"      text length: {len(metadata['text'])}")

# Step 6: Check existing embeddings metadata
print(f"\n📊 Checking existing embeddings metadata...")

for emb_type in ['naive_embedding', 'tfidf_embedding', 'prefix_fusion_embedding']:
    emb_meta_path = f"embeddings_output/semantic/{emb_type}/metadata.json"
    if os.path.exists(emb_meta_path):
        print(f"\n   ✅ Found: {emb_type}/metadata.json")
        
        try:
            with open(emb_meta_path, 'r', encoding='utf-8') as f:
                emb_meta = json.load(f)
            
            if emb_meta:
                first_key = list(emb_meta.keys())[0]
                first_entry = emb_meta[first_key]
                print(f"      Keys in first entry: {list(first_entry.keys())}")
                print(f"      document_name present: {'document_name' in first_entry}")
                if 'document_name' in first_entry:
                    print(f"      document_name value: '{first_entry['document_name']}'")
        except Exception as e:
            print(f"      ❌ Error reading: {str(e)}")
    else:
        print(f"   ❌ Not found: {emb_type}/metadata.json")

print("\n" + "=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)