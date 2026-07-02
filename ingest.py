import csv, uuid, pathlib
from qdrant_client import QdrantClient, models
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm
import config as C

client = QdrantClient(url=C.QDRANT_URL)
embedder = OllamaEmbeddings(model=C.EMBED_MODEL, base_url=C.EMBED_BASE_URL)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=C.CHUNK_SIZE, chunk_overlap=C.CHUNK_OVERLAP
)

def load_manifest():
    with open("manifest.csv") as f:
        return {
            r["filename"].strip(): r["classification"].strip().lower()
            for r in csv.DictReader(f)
        }

def main():
    client.recreate_collection(
        collection_name=C.COLLECTION,
        vectors_config=models.VectorParams(size=C.EMBED_DIM, distance=models.Distance.COSINE),
    )
    # Index the governance field so payload filtering is fast.
    client.create_payload_index(
        collection_name=C.COLLECTION,
        field_name="classification_rank",
        field_schema=models.PayloadSchemaType.INTEGER,
    )

    manifest = load_manifest()
    points = []
    n_docs = 0
    for path in tqdm(sorted(pathlib.Path("documents").glob("*")), desc="documents"):
        if path.suffix not in (".md", ".txt"):
            continue
        classification = manifest.get(path.name, "restricted")  # fail-closed default
        if classification not in C.CLASSIFICATION_RANK:
            # Unknown label in the manifest: fail closed, don't crash or fail open.
            print(f"⚠️  {path.name}: unknown classification {classification!r} — treating as restricted")
            classification = "restricted"
        rank = C.CLASSIFICATION_RANK[classification]
        n_docs += 1
        chunks = splitter.split_text(path.read_text(encoding="utf-8"))
        vectors = embedder.embed_documents(chunks)
        for chunk, vec in zip(chunks, vectors):
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "text": chunk,
                    "source": path.name,
                    "classification": classification,
                    "classification_rank": rank,
                },
            ))
    client.upsert(collection_name=C.COLLECTION, points=points)
    print(f"Indexed {len(points)} chunks from {n_docs} documents.")

if __name__ == "__main__":
    main()
