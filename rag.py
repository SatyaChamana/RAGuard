import time
from langchain_ollama import OllamaEmbeddings, ChatOllama
import config as C
from governance import clearance_filter, make_client

client = make_client()
embedder = OllamaEmbeddings(model=C.EMBED_MODEL, base_url=C.EMBED_BASE_URL)
llm = ChatOllama(model=C.GEN_MODEL, temperature=0, base_url=C.GEN_BASE_URL)

def authorized_search(query: str, user_clearance: int, top_k: int = C.TOP_K):
    """Retrieval WITH the governance gate baked in.
    Docs above the user's clearance never enter the candidate set."""
    qvec = embedder.embed_query(query)
    t0 = time.perf_counter()
    hits = client.query_points(
        collection_name=C.COLLECTION,
        query=qvec,
        limit=top_k,
        query_filter=clearance_filter(user_clearance),  # doc_rank <= user_clearance
        with_payload=True,
    ).points
    latency_ms = (time.perf_counter() - t0) * 1000
    return hits, latency_ms

PROMPT = """Answer the question using ONLY the context below. \
If the context doesn't contain the answer, say "I don't have that information."

Context:
{context}

Question: {question}
Answer:"""

def answer(query: str, user_clearance: int):
    hits, ret_ms = authorized_search(query, user_clearance)
    contexts = [h.payload["text"] for h in hits]
    sources = [h.payload["source"] for h in hits]
    t0 = time.perf_counter()
    if contexts:
        resp_content = llm.invoke(
            PROMPT.format(context="\n\n".join(contexts), question=query)
        ).content
    else:
        # No permitted context: answer deterministically instead of letting the
        # model free-wheel on an empty prompt (it could answer from pretraining).
        resp_content = "I don't have that information."
    gen_ms = (time.perf_counter() - t0) * 1000
    return {
        "question": query,
        "answer": resp_content,
        "retrieved_contexts": contexts,
        "sources": sources,
        "retrieval_ms": ret_ms,
        "generation_ms": gen_ms,
    }

if __name__ == "__main__":
    out = answer("What does the handbook say about remote work?", user_clearance=2)
    print(out["answer"], "\n\nsources:", out["sources"])
