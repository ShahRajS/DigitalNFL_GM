import os
import sys
from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone

# Ensure script directory is in path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from rag_indexer import index_draft_packet, add_custom_snippet_to_db, get_genai_client, get_embeddings_with_retry

# Load env variables from root
dotenv_path = os.path.join(os.path.dirname(script_dir), '.env')
load_dotenv(dotenv_path)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def load_vector_db() -> list[dict]:
    """Verifies Pinecone index status on startup. Auto-indexes PDF if index is missing or empty."""
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set in environment variables.")
        
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "digital-nfl-gm"
    
    existing_indexes = [i.name for i in pc.list_indexes()]
    pdf_file = os.path.join(script_dir, "2025-San-Francisco-49ers-Draft-Packet.pdf")
    
    # Auto-index if not found
    if index_name not in existing_indexes:
        print(f"Pinecone index '{index_name}' not found. Starting auto-indexing process...")
        if os.path.exists(pdf_file):
            index_draft_packet(pdf_file, index_name)
        else:
            raise FileNotFoundError(
                f"Could not load vector DB. Pinecone index was missing and source PDF was not found at {pdf_file}."
            )
    else:
        # Check if empty
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        if stats.get("total_vector_count", 0) == 0:
            print(f"Pinecone index '{index_name}' is empty. Starting auto-indexing process...")
            if os.path.exists(pdf_file):
                index_draft_packet(pdf_file, index_name)
            else:
                raise FileNotFoundError(
                    f"Could not load vector DB. Pinecone index was empty and source PDF was not found at {pdf_file}."
                )
                
    print(f"Verified Pinecone vector database '{index_name}' is online and indexed.")
    return [] # Return empty list, since we query Pinecone directly now

def retrieve_chunks(query: str, top_k: int = 3) -> list[dict]:
    """Retrieves the top_k most relevant chunks matching the query string from Pinecone.
    
    Args:
        query: The search query.
        top_k: Number of matching chunks to return.
        
    Returns:
        List of chunks with similarity score added:
        [{"text": "...", "metadata": {...}, "score": 0.85}]
    """
    load_vector_db() # Ensure connection is verified and populated
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("digital-nfl-gm")
    
    client = get_genai_client()
    
    # Generate query embedding using the retry helper
    embeddings = get_embeddings_with_retry(client, [query])
    query_vector = embeddings[0]
    
    # Query Pinecone index
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    scored_chunks = []
    for match in response.matches:
        meta = match.metadata
        text = meta.get("text", "")
        # Format metadata to look like the local cached structure for ADK compatibility
        formatted_meta = {
            "source": meta.get("source", "Unknown"),
            "page": int(meta.get("page", 0)),
            "date_added": meta.get("date_added", "")
        }
        scored_chunks.append({
            "text": text,
            "metadata": formatted_meta,
            "score": match.score
        })
        
    return scored_chunks

def add_scout_note(text: str, source: str = "Scout Note", metadata: dict = None) -> str:
    """Public API to append a custom snippet directly to the Pinecone vector index.
    
    Args:
        text: Snippet text content.
        source: Name of the source (e.g. 'GM Notes', 'Combine').
        metadata: Optional dictionary of attributes.
    Returns:
        A confirmation message.
    """
    success = add_custom_snippet_to_db(text, source, metadata)
    if success:
        return f"Successfully added note from '{source}' to the Pinecone cloud index."
    else:
        return f"Failed to add note from '{source}' to Pinecone."

if __name__ == "__main__":
    # Test RAG retrieval directly if run as main
    print("Testing Pinecone Vector Database retrieval...")
    try:
        results = retrieve_chunks("What are the draft dates?", top_k=2)
        for i, res in enumerate(results):
            print(f"\nMatch {i+1} (Score: {res['score']:.4f}) - Source: {res['metadata']['source']} [Page {res['metadata'].get('page', 'N/A')}]:")
            print("-" * 50)
            print(res['text'][:300] + "...")
            print("-" * 50)
    except Exception as e:
        print(f"Error during Pinecone retrieval test: {e}")
