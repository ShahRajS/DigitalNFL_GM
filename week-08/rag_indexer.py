import os
import json
import uuid
import time
from datetime import datetime
from dotenv import load_dotenv
from google import genai
import PyPDF2
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path)

# Map key if GEMINI_API_KEY is not set but GOOGLE_API_KEY is
if "GOOGLE_API_KEY" in os.environ and "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]

def get_genai_client():
    """Initializes and returns the Google GenAI Client."""
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment variables.")
    return genai.Client()

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extracts text page-by-page from a PDF file.
    
    Returns:
        A list of dicts, where each dict contains page text and the 1-based page number:
        [{"text": "...", "page": 1}]
    """
    print(f"Reading PDF from: {pdf_path}")
    pages = []
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            # Clean curly braces to prevent ADK parser KeyError issues down the line
            text = text.replace("{", "[").replace("}", "]")
            pages.append({
                "text": text,
                "page": idx + 1
            })
    print(f"Successfully extracted {len(pages)} pages.")
    return pages

def chunk_text(pages: list[dict], source_name: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[dict]:
    """Splits page text into smaller, overlapping chunks to keep paragraphs/sentences intact.
    
    Args:
        pages: List of page dicts.
        source_name: The name of the source PDF.
        chunk_size: Maximum character count of a single chunk.
        chunk_overlap: Overlapped character count between consecutive chunks.
        
    Returns:
        A list of chunk dictionaries containing text and metadata:
        [{"text": "...", "metadata": {"source": "filename", "page": 1}}]
    """
    print(f"Splitting pages into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    chunks = []
    
    for page_data in pages:
        text = page_data["text"]
        page_num = page_data["page"]
        
        # Simple sliding window chunker
        start = 0
        text_length = len(text)
        
        # If the page is empty, skip
        if text_length == 0:
            continue
            
        # If the entire page is smaller than the chunk size, add it as a single chunk
        if text_length <= chunk_size:
            chunks.append({
                "text": text,
                "metadata": {
                    "source": source_name,
                    "page": page_num
                }
            })
            continue
            
        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            # Extract current chunk window
            chunk_content = text[start:end]
            
            # Avoid cutting words in the middle by shifting back to the nearest space
            # if we are not at the end of the text
            if end < text_length:
                last_space = chunk_content.rfind(" ")
                if last_space > (chunk_size // 2):  # only back off if we don't truncate too much
                    end = start + last_space
                    chunk_content = text[start:end]
            
            chunks.append({
                "text": chunk_content.strip(),
                "metadata": {
                    "source": source_name,
                    "page": page_num
                }
            })
            
            if end == text_length:
                break
                
            # Advance start pointer by (chunk_size - overlap)
            start = end - chunk_overlap
            if start >= text_length:
                break
                
    print(f"Generated {len(chunks)} chunks.")
    return chunks

def get_embeddings_with_retry(client, contents: list[str], model: str = "models/gemini-embedding-2") -> list[list[float]]:
    """Helper function to fetch embeddings from the Gemini API with automatic exponential backoff on 429 limits."""
    import time
    max_attempts = 6
    for attempt in range(max_attempts):
        try:
            response = client.models.embed_content(
                model=model,
                contents=contents
            )
            return [emb.values for emb in response.embeddings]
        except Exception as e:
            if "429" in str(e) and attempt < max_attempts - 1:
                sleep_time = 20 * (attempt + 1)
                print(f"Embedding API rate limited (429). Retrying in {sleep_time}s... (Attempt {attempt+1}/{max_attempts})")
                time.sleep(sleep_time)
            else:
                raise e
    raise RuntimeError("Failed to generate embeddings after multiple retries.")

def generate_embeddings_for_chunks(client, chunks: list[dict]) -> list[dict]:
    """Calls Google GenAI API to generate vector embeddings for each chunk.
    
    Uses `models/gemini-embedding-2` for 3072-dimensional embeddings with automatic retry on rate limits.
    """
    import time
    print(f"Generating embeddings for {len(chunks)} chunks using models/gemini-embedding-2...")
    embedded_chunks = []
    
    batch_size = 30
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [c["text"] for c in batch]
        
        embeddings = get_embeddings_with_retry(client, texts)
            
        for idx, item in enumerate(batch):
            embedded_chunks.append({
                "id": str(uuid.uuid4()),
                "text": item["text"],
                "metadata": item["metadata"],
                "embedding": embeddings[idx]
            })
        print(f"Indexed batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} ({len(embedded_chunks)}/{len(chunks)} chunks)")
        
        # Pause briefly between batches to respect rate limits
        if i + batch_size < len(chunks):
            time.sleep(2)
            
    return embedded_chunks

def add_custom_snippet_to_db(text: str, source_name: str, metadata: dict = None) -> bool:
    """Utility function allowing the user to append custom text/notes directly to Pinecone.
    
    Args:
        text: The snippet content to add.
        source_name: A name identifying the source of this info (e.g. 'Scout Notes', 'Combine Results').
        metadata: Any optional dictionary of extra attributes.
        
    Returns:
        True if successful.
    """
    try:
        client = get_genai_client()
        pinecone_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_key:
            raise ValueError("PINECONE_API_KEY is not set in environment variables.")
            
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index("digital-nfl-gm")
        
        # Generate embedding for the new text using the retry helper
        embeddings = get_embeddings_with_retry(client, [text])
        embedding = embeddings[0]
        
        # Build metadata
        meta = {
            "text": text.strip(),
            "source": source_name,
            "date_added": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "page": 0
        }
        if metadata:
            meta.update(metadata)
            
        vector_id = str(uuid.uuid4())
        index.upsert(vectors=[{
            "id": vector_id,
            "values": embedding,
            "metadata": meta
        }])
        
        print(f"Successfully added custom snippet from '{source_name}' to Pinecone.")
        return True
    except Exception as e:
        print(f"Failed to add snippet to Pinecone: {e}")
        return False

def index_draft_packet(pdf_path: str, index_name: str = "digital-nfl-gm"):
    """Orchestrates reading, chunking, embedding, and uploading the Draft Packet to Pinecone with resume capability."""
    client = get_genai_client()
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        raise ValueError("PINECONE_API_KEY is not set in environment variables.")
        
    # Connect to Pinecone and create index if missing
    pc = Pinecone(api_key=pinecone_key)
    existing_indexes = [i.name for i in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Creating Pinecone index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Waiting for index to initialize...")
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(2)
        print("Index ready!")
        
    index = pc.Index(index_name)
    
    pages = extract_text_from_pdf(pdf_path)
    source_name = os.path.basename(pdf_path)
    chunks = chunk_text(pages, source_name)
    
    # Check what is already indexed in Pinecone to support resumption
    print(f"Checking index status in Pinecone for source '{source_name}'...")
    try:
        response = index.query(
            vector=[0.0] * 3072,
            filter={"source": source_name},
            top_k=10000,
            include_metadata=True
        )
        already_indexed_texts = {m.metadata.get("text", "").strip() for m in response.matches if m.metadata}
        print(f"Found {len(already_indexed_texts)} chunks already indexed in Pinecone for this document.")
    except Exception as e:
        print(f"Warning/Error fetching existing vectors from Pinecone: {e}. Starting indexing from scratch.")
        already_indexed_texts = set()
        
    # Filter local chunks to only index what is missing
    chunks_to_index = [c for c in chunks if c["text"].strip() not in already_indexed_texts]
    print(f"Remaining chunks to process: {len(chunks_to_index)} / {len(chunks)}")
    
    if len(chunks_to_index) == 0:
        print("All chunks are already indexed! No new embeddings needed.")
        return
        
    print(f"Generating embeddings and uploading {len(chunks_to_index)} chunks in batches of 30...")
    batch_size = 30
    total_uploaded = len(already_indexed_texts)
    
    for i in range(0, len(chunks_to_index), batch_size):
        batch = chunks_to_index[i:i+batch_size]
        texts = [c["text"] for c in batch]
        
        # Call Google GenAI Embeddings API with automatic rate-limit retries
        embeddings = get_embeddings_with_retry(client, texts)
        
        # Format vectors for Pinecone
        pinecone_vectors = []
        for idx, item in enumerate(batch):
            metadata = {
                "text": item["text"],
                "source": item["metadata"]["source"],
                "page": int(item["metadata"].get("page", 0)),
                "date_added": item["metadata"].get("date_added", "")
            }
            pinecone_vectors.append({
                "id": str(uuid.uuid4()),
                "values": embeddings[idx],
                "metadata": metadata
            })
            
        # Upsert immediately to Pinecone
        index.upsert(vectors=pinecone_vectors)
        total_uploaded += len(batch)
        print(f"Indexed & Uploaded batch {i//batch_size + 1}/{(len(chunks_to_index)-1)//batch_size + 1} ({total_uploaded}/{len(chunks)} total chunks in Pinecone)")
        
        # Pause briefly to respect rate limits
        if i + batch_size < len(chunks_to_index):
            time.sleep(2)
            
    print(f"Successfully processed all chunks for '{source_name}' in Pinecone index '{index_name}'.")

if __name__ == "__main__":
    # If run directly, index the local PDF
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_file = os.path.join(script_dir, "2025-San-Francisco-49ers-Draft-Packet.pdf")
    
    if os.path.exists(pdf_file):
        index_draft_packet(pdf_file)
    else:
        print(f"Could not locate PDF at {pdf_file}. Make sure it is copied to week-08.")
