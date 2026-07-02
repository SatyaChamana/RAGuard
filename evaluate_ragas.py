import json
from ragas import evaluate, EvaluationDataset
from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness, ResponseRelevancy,
    LLMContextPrecisionWithReference, LLMContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings
from rag import answer
import config as C

judge = LangchainLLMWrapper(ChatOllama(model=C.JUDGE_MODEL, temperature=0, base_url=C.JUDGE_BASE_URL))
judge_emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=C.EMBED_MODEL, base_url=C.EMBED_BASE_URL))

def build_samples():
    golden = json.load(open("golden_set.json"))
    rows = []
    for item in golden:
        # Run each question at MAX clearance so retrieval is fully tested
        # (governance is tested separately in test_governance.py).
        out = answer(item["question"], user_clearance=2)
        rows.append({
            "user_input": item["question"],
            "retrieved_contexts": out["retrieved_contexts"],
            "response": out["answer"],
            "reference": item["ground_truth"],
        })
    return EvaluationDataset.from_list(rows)

def main():
    dataset = build_samples()
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),                          # answer grounded in context?
            ResponseRelevancy(),                     # answer addresses the question?
            LLMContextPrecisionWithReference(),      # retrieved context relevant?
            LLMContextRecall(),                      # did we retrieve all needed context?
        ],
        llm=judge,
        embeddings=judge_emb,
        # A single heavy judge (e.g. 70B) on one GPU can't serve 16 concurrent
        # requests — they queue and blow the default 180s timeout, yielding NaN
        # scores. Throttle concurrency and extend the timeout so every job lands.
        run_config=RunConfig(timeout=600, max_workers=4),
    )
    print(result)
    result.to_pandas().to_csv("ragas_results.csv", index=False)
    print("\nPer-question detail written to ragas_results.csv")

if __name__ == "__main__":
    main()
