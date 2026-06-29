import os
import json
import uuid
import time
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types
import pypdf
import fitz
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path)

# Map key if GEMINI_API_KEY is not set but GOOGLE_API_KEY is
if "GOOGLE_API_KEY" in os.environ and "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]

_current_key_idx = 0

def get_api_keys() -> list[str]:
    pool_str = os.getenv("GEMINI_API_KEY_POOL")
    if pool_str:
        keys = pool_str.split(",")
    else:
        # Fallback to single keys
        keys = [
            os.getenv("GOOGLE_API_KEY"),
            os.getenv("GEMINI_API_KEY")
        ]
    return [k.strip() for k in keys if k and k.strip()]

def get_genai_client():
    """Initializes and returns the Google GenAI Client using the current rotated key."""
    global _current_key_idx
    keys = get_api_keys()
    if not keys:
        raise ValueError("No API keys found in environment or key pool.")
    selected_key = keys[_current_key_idx % len(keys)]
    os.environ["GEMINI_API_KEY"] = selected_key
    os.environ["GOOGLE_API_KEY"] = selected_key
    return genai.Client(api_key=selected_key)

def rotate_genai_key():
    """Rotates to the next API key in the pool."""
    global _current_key_idx
    keys = get_api_keys()
    _current_key_idx = (_current_key_idx + 1) % len(keys)
    print(f"\n[Key Rotator] Rotating to API key slot {_current_key_idx+1}/{len(keys)}...")

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extracts text page-by-page from a PDF file using pypdf.
    
    Returns:
        A list of dicts, where each dict contains page text and the 1-based page number:
        [{"text": "...", "page": 1}]
    """
    print(f"Reading PDF from: {pdf_path}")
    pages = []
    with open(pdf_path, 'rb') as f:
        reader = pypdf.PdfReader(f)
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

def extract_multimodal_pages_from_pdf(pdf_path: str) -> list[dict]:
    """Converts PDF pages to images and uses Gemini 2.5 Flash to extract layout-aware markdown and visual descriptions.
    
    Returns:
        A list of dicts, where each dict contains page text and the 1-based page number:
        [{"text": "...", "page": 1}]
    """
    cache_path = pdf_path + ".multimodal_cache.json"
    pages = []
    if os.path.exists(cache_path):
        print(f"Loading multimodal extraction cache from: {cache_path}")
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                pages = json.load(f)
            print(f"Loaded {len(pages)} previously extracted pages from cache.")
        except Exception as e:
            print(f"Warning: Failed to load cache from {cache_path}: {e}. Proceeding with empty cache.")
            pages = []

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    # Check if all pages are already cached
    if len(pages) >= total_pages:
        print(f"All {total_pages} pages are already in the multimodal cache.")
        doc.close()
        return pages

    print(f"Reading PDF multimodally from: {pdf_path}")
    client = get_genai_client()
    
    # Prompt for multimodal layout/table extraction and image descriptions
    prompt = (
        "You are an expert document parser. Convert this PDF page image into clean, structured Markdown.\n"
        "1. Identify any tables and represent them exactly as Markdown tables (e.g. using | Col 1 | Col 2 |).\n"
        "2. If there are any charts, graphs, diagrams, or key photos, write a detailed text description summarizing the content, "
        "trends, axes, labels, and takeaways. Embed this description directly under the relevant heading.\n"
        "3. Preserve the layout hierarchical headings (#, ##, ###) for standard text blocks.\n"
        "4. Do not include raw page numbers or headers/footers if they are repetitive."
    )
    
    for idx, page in enumerate(doc):
        page_num = idx + 1
        
        # Check if already cached
        if any(p["page"] == page_num for p in pages):
            continue
            
        print(f"Processing page {page_num}/{total_pages} multimodally...")
        
        # Render page to PNG image bytes (dpi=150 is ideal for clarity and token limits)
        pix = page.get_pixmap(dpi=150)
        image_bytes = pix.tobytes("png")
        
        # Call Gemini model with retry on rate limits (429) or temporary server errors
        max_attempts = 6
        page_text = ""
        
        for attempt in range(max_attempts):
            try:
                client = get_genai_client()
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type="image/png"
                        ),
                        prompt
                    ]
                )
                page_text = response.text or ""
                break
            except Exception as e:
                # If we hit a rate limit (429) or quota limit, rotate the key and retry
                if "429" in str(e) or "quota" in str(e).lower() or "limit" in str(e).lower() or "exhausted" in str(e).lower():
                    # If we have rotated through all keys and still exhausted, raise error and halt
                    if attempt == max_attempts - 1:
                        print(f"\n[Error] All API keys in the pool are exhausted at page {page_num}. Aborting indexing.")
                        raise RuntimeError(f"All API keys exhausted. Page {page_num} could not be parsed multimodally.")
                    else:
                        print(f"Gemini API rate limit/quota reached on page {page_num} (Attempt {attempt+1}/{max_attempts}). Rotating key...")
                        rotate_genai_key()
                        time.sleep(2) # Brief pause before retry
                elif ("500" in str(e) or "503" in str(e)) and attempt < max_attempts - 1:
                    sleep_time = 15 * (attempt + 1)
                    print(f"Gemini API temporary error on page {page_num}. Retrying in {sleep_time}s... (Attempt {attempt+1}/{max_attempts})")
                    time.sleep(sleep_time)
                else:
                    print(f"Failed to process page {page_num} multimodally: {e}")
                    raise e
                    
        # Replace curly braces to prevent ADK parser KeyError issues down the line
        page_text = page_text.replace("{", "[").replace("}", "]")
        
        pages.append({
            "text": page_text,
            "page": page_num
        })
        
        # Save cache page-by-page
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(pages, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
            
        # Pacing to respect Google AI Studio free tier limits
        if page_num < total_pages:
            time.sleep(6)
            
    doc.close()
    print(f"Successfully extracted {len(pages)} pages multimodally.")
    return pages

def chunk_text(pages: list[dict], source_name: str, chunk_size: int = 2000, chunk_overlap: int = 300) -> list[dict]:
    """Splits page text into paragraph-based chunks, maintaining paragraph integrity
    and mapping chunk start character index back to its 1-based page number.
    """
    print(f"Splitting pages into paragraph chunks (size={chunk_size}, overlap={chunk_overlap})...")
    
    # 1. Concatenate all pages into a single full_text while tracking character offset to page mappings
    full_text = ""
    page_ranges = [] # list of (start_idx, end_idx, page_num)
    
    for page_data in pages:
        p_text = page_data["text"]
        p_num = page_data["page"]
        start_idx = len(full_text)
        full_text += p_text + "\n\n"
        end_idx = len(full_text)
        page_ranges.append((start_idx, end_idx, p_num))
        
    def get_page_for_char_index(char_idx: int) -> int:
        for start, end, p_num in page_ranges:
            if start <= char_idx < end:
                return p_num
        return page_ranges[-1][2] if page_ranges else 1

    # 2. Split full_text into paragraphs, keeping track of their start character index
    paragraphs = []
    current_idx = 0
    raw_paragraphs = full_text.split("\n\n")
    for rp in raw_paragraphs:
        rp_len = len(rp)
        if rp_len == 0:
            current_idx += 2 # length of "\n\n"
            continue
        paragraphs.append({
            "text": rp.strip(),
            "start_idx": current_idx,
            "length": rp_len
        })
        current_idx += rp_len + 2 # paragraph text + "\n\n"

    # Filter out empty or whitespace-only paragraphs
    paragraphs = [p for p in paragraphs if p["text"]]
    
    # Split any paragraph that is larger than chunk_size into smaller sub-paragraphs (e.g. line by line)
    split_paragraphs = []
    for p in paragraphs:
        p_text = p["text"]
        p_start = p["start_idx"]
        
        if len(p_text) <= chunk_size:
            split_paragraphs.append(p)
        else:
            lines = p_text.split("\n")
            sub_para_text = ""
            sub_para_start = p_start
            
            for line in lines:
                if len(sub_para_text) + len(line) + 1 <= chunk_size or not sub_para_text:
                    if sub_para_text:
                        sub_para_text += "\n" + line
                    else:
                        sub_para_text = line
                else:
                    split_paragraphs.append({
                        "text": sub_para_text.strip(),
                        "start_idx": sub_para_start,
                        "length": len(sub_para_text)
                    })
                    sub_para_start = p_start + p_text.find(line)
                    sub_para_text = line
            
            if sub_para_text:
                split_paragraphs.append({
                    "text": sub_para_text.strip(),
                    "start_idx": sub_para_start,
                    "length": len(sub_para_text)
                })
                
    paragraphs = split_paragraphs
    
    # 3. Assemble paragraphs into chunks
    chunks = []
    
    # If there are no paragraphs, return empty
    if not paragraphs:
        return chunks
        
    i = 0
    num_paras = len(paragraphs)
    
    while i < num_paras:
        current_chunk_paras = []
        current_chunk_len = 0
        
        # Add paragraphs to current chunk until we reach chunk_size
        j = i
        while j < num_paras:
            para = paragraphs[j]
            para_len = len(para["text"])
            
            # If adding this paragraph exceeds chunk_size and we already have at least one paragraph in the chunk, stop.
            if current_chunk_len + para_len > chunk_size and current_chunk_len > 0:
                break
                
            current_chunk_paras.append(para)
            current_chunk_len += para_len + 2 # text + spacer
            j += 1
            
        # Combine the paragraphs into a single text chunk
        chunk_text_str = "\n\n".join([p["text"] for p in current_chunk_paras])
        
        # Determine the page number. We use the page of the start of the first paragraph in this chunk.
        start_char_idx = current_chunk_paras[0]["start_idx"]
        page_num = get_page_for_char_index(start_char_idx)
        
        chunks.append({
            "text": chunk_text_str.strip(),
            "metadata": {
                "source": source_name,
                "page": page_num
            }
        })
        
        # If we have reached the end of all paragraphs, we are done
        if j == num_paras:
            break
            
        # Determine overlap
        overlap_len = 0
        step_back = 0
        while j - step_back - 1 > i:
            prev_para = paragraphs[j - step_back - 1]
            overlap_len += len(prev_para["text"]) + 2
            if overlap_len > chunk_overlap:
                step_back += 1
                break
            step_back += 1
            
        i = j - step_back if step_back > 0 else j
        
    print(f"Generated {len(chunks)} paragraph chunks.")
    return chunks

def get_embeddings_with_retry(contents: list[str], model: str = "models/gemini-embedding-2") -> list[list[float]]:
    """Helper function to fetch embeddings from the Gemini API with key rotation on 429 limits."""
    import time
    max_attempts = 6
    for attempt in range(max_attempts):
        try:
            client = get_genai_client()
            response = client.models.embed_content(
                model=model,
                contents=contents
            )
            return [emb.values for emb in response.embeddings]
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower() or "limit" in str(e).lower() or "exhausted" in str(e).lower():
                if attempt == max_attempts - 1:
                    raise e
                print(f"Embedding API rate limited (429). Rotating key... (Attempt {attempt+1}/{max_attempts})")
                rotate_genai_key()
                time.sleep(2) # Brief pause before retry
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
        embeddings = get_embeddings_with_retry([text])
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

def index_draft_packet(pdf_path: str, index_name: str = "digital-nfl-gm", multimodal: bool = True):
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
    
    source_name = os.path.basename(pdf_path)
    quota_exhausted_during_extraction = False
    
    if multimodal:
        try:
            pages = extract_multimodal_pages_from_pdf(pdf_path)
        except Exception as e:
            if "All API keys exhausted" in str(e):
                print(f"\n[Notice] Quota exhausted during multimodal extraction. Proceeding to index pages extracted so far.")
                quota_exhausted_during_extraction = True
                # Load pages from cache file since exception occurred midway
                cache_path = pdf_path + ".multimodal_cache.json"
                if os.path.exists(cache_path):
                    with open(cache_path, "r", encoding="utf-8") as f:
                        pages = json.load(f)
                else:
                    pages = []
            else:
                raise e
        # Use larger chunk size for markdown pages to preserve table layouts
        chunks = chunk_text(pages, source_name, chunk_size=2000, chunk_overlap=300)
    else:
        pages = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(pages, source_name, chunk_size=1000, chunk_overlap=200)
    
    # Check what is already indexed in Pinecone to support resumption
    print(f"Checking index status in Pinecone for source '{source_name}'...")
    try:
        response = index.query(
            vector=[0.0] * 3072,
            filter={"source": source_name},
            top_k=10000,
            include_metadata=True
        )
        already_indexed_texts = set()
        for m in response.matches:
            meta = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", None)
            if meta:
                text = meta.get("text", "")
                if text:
                    already_indexed_texts.add(text.strip())
        print(f"Found {len(already_indexed_texts)} chunks already indexed in Pinecone for this document.")
    except Exception as e:
        print(f"Warning/Error fetching existing vectors from Pinecone: {e}. Starting indexing from scratch.")
        already_indexed_texts = set()
        
    # Filter local chunks to only index what is missing
    chunks_to_index = [c for c in chunks if c["text"].strip() not in already_indexed_texts]
    print(f"Remaining chunks to process: {len(chunks_to_index)} / {len(chunks)}")
    
    if len(chunks_to_index) == 0:
        print("All chunks are already indexed! No new embeddings needed.")
        if quota_exhausted_during_extraction:
            print(f"\n[Error] Halting execution: PDF '{source_name}' was only partially indexed due to quota exhaustion.")
            raise RuntimeError(f"All API keys exhausted. PDF '{source_name}' was only partially indexed.")
        return
        
    print(f"Generating embeddings and uploading {len(chunks_to_index)} chunks in batches of 30...")
    batch_size = 30
    total_uploaded = len(already_indexed_texts)
    
    for i in range(0, len(chunks_to_index), batch_size):
        batch = chunks_to_index[i:i+batch_size]
        texts = [c["text"] for c in batch]
        
        # Call Google GenAI Embeddings API with automatic rate-limit retries
        embeddings = get_embeddings_with_retry(texts)
        
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
    
    if quota_exhausted_during_extraction:
        print(f"\n[Error] Halting execution: PDF '{source_name}' was only partially indexed due to quota exhaustion.")
        raise RuntimeError(f"All API keys exhausted. PDF '{source_name}' was only partially indexed.")
 
if __name__ == "__main__":
    # If run directly, index all local target PDFs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_files = [
        "2026-San-Francisco-49ers-Draft-Packet.pdf",
        "2025-San-Francisco-49ers-Draft-Packet.pdf",
        "San-Francisco-49ers-2025-Season-Review.pdf"
    ]
    
    indexed_any = False
    for filename in target_files:
        pdf_path = os.path.join(script_dir, filename)
        if os.path.exists(pdf_path):
            print(f"\n--- Indexing {filename} ---")
            index_draft_packet(pdf_path, multimodal=False)
            indexed_any = True
        else:
            print(f"Could not locate PDF at {pdf_path}. Skipping.")
            
    if not indexed_any:
        print("Error: No target PDFs found in week-09 directory to index.")
