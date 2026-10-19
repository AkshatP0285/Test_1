# RAG: Baseline Retrieval vs Cross-Encoder Re-Ranking (D&D 5e SRD)

Two-stage retrieval over the D&D 5e System Reference Document 5.1:

1. **Pipeline 1 (baseline):** bi-encoder (BAAI/bge-small-en-v1.5) + FAISS, top 10, the first 3 chunks go to the LLM.
2. **Pipeline 2 (re-ranking):** the same 10 candidates re-scored by a cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2), the top 3 go to the LLM.

Both pipelines use the same chunks and the same prompt.

## Main notebook
`rag_assignment.ipynb` (executed, outputs included) contains the whole assignment: PDF loading and chunking, both pipelines,
8 failure-mode queries with rank shifts and the two LLM answers side by side, and a latency profile.
The LLM is Gemini (`gemini-3.5-flash-lite`). Answers are cached, so a re-run costs no API quota.

## Data
The document used is the D&D 5e System Reference Document 5.1, included as `data/SRD_CC_v5.1.pdf` (403 pages, two columns).
The chunks retrieved for every question are saved in readable form in `outputs_25q/retrieved_chunks.md`: for each of the 25 questions
and both embedders, the ten bi-encoder results, their order after the cross-encoder, scores, page numbers and the chunk text.
`outputs_25q/retrieval.json` holds the same ids plus all 1,504 chunks.

## Where the models are
| Role | Model | Notebook / code |
|---|---|---|
| Bi-encoder | `BAAI/bge-small-en-v1.5` | `rag_assignment.ipynb` (section 2, `retrieve_top10`), also `rag_retrieval_25q.ipynb` |
| Bi-encoder | `google/embeddinggemma-2` | `rag_retrieval_25q.ipynb` and `rag_experiments.ipynb` only (`EMBEDDERS` dictionary) |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `rag_assignment.ipynb` (section 2, `rerank`), and the two notebooks above |
| LLM | Gemini `gemini-3.5-flash-lite` | `rag_assignment.ipynb` (`ask_llm`) |
| LLM | `Qwen/Qwen3-4B`, `Qwen/Qwen3.5-9B` | `rag_experiments.ipynb`, `rag_generation_qwen35_9b.ipynb` |

## Run it
```
pip install -r requirements.txt
export GEMINI_API_KEY=...      # never commit this
# the PDF is in data/; set the PDF / OUT paths in the first code cell (the notebooks were run with cluster paths), then run the notebook
```

## Optional extra notebooks
* `rag_experiments.ipynb`: two embedders (bge-small, EmbeddingGemma 2 in float32) with a local Qwen3-4B generator, on the 8 queries.
  EmbeddingGemma 2 also needs `pillow` and `torchvision` because its processor is multimodal.
* `rag_retrieval_25q.ipynb`: recall@3, recall@10 and MRR on 25 questions for both embedders, with and without the cross-encoder.
* `rag_generation_qwen35_9b.ipynb`: answers from a local Qwen3.5-9B split over two 11 GB GPUs (fp16, thinking off).

## Results
* `outputs/`: results of the main notebook (`rank_shift.csv`, `latency.csv`, `results.json`) and `run1_gemini-3.8-flash/`, an earlier run on another Gemini model.
* `outputs_exp/`: the Qwen3-4B experiment.
* `outputs_25q/`: the 25-question study. `retrieval.json` holds the retrieved chunk ids used by the generation notebook,
  `generations_with_thinking_partial.json` is a thinking-on attempt that ran out of tokens, and `gold_phrase_check.txt` shows
  the passage each gold phrase matches.

## Attribution
This work includes material taken from the System Reference Document 5.1 ("SRD 5.1") by Wizards of the Coast LLC and available at
https://dnd.wizards.com/resources/systems-reference-document. The SRD 5.1 is licensed under the Creative Commons Attribution 4.0
International License available at https://creativecommons.org/licenses/by/4.0/legalcode.
