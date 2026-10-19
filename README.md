# RAG: baseline retrieval vs cross-encoder re-ranking (D&D 5e SRD, Qwen3.5-9B, 25 questions)

Everything is in one notebook, `rag_assignment.ipynb` (executed, outputs included).

## What it does
Two retrieval workflows over the **same chunked document** (the D&D 5e System Reference Document 5.1, `data/SRD_CC_v5.1.pdf`):

1. **Pipeline 1, baseline:** a bi-encoder embeds every chunk into a FAISS inner-product index; for each question the top 10 chunks are retrieved and the first 3 go to the LLM, which gives **Answer 1**.
2. **Pipeline 2, re-ranking:** the same 10 candidates are scored by a cross-encoder, re-sorted, and the first 3 go to the LLM with the identical prompt, which gives **Answer 2**.

It is run for 25 questions and for two bi-encoders, and the answers come from a local Qwen3.5-9B model.

## Flow of the notebook
| Section | Content |
|---|---|
| 1 | Load the PDF (two columns, read one column at a time) and cut it into 1,504 chunks of 200 words with a 40-word overlap |
| 2 | The 25 questions and, for each, the SRD passage that answers it (chosen by hand, tied to the chunks by a short quote); the matching passage is printed so every choice can be checked |
| 3 | Stage 1 (bi-encoder + FAISS, top 10) and stage 2 (cross-encoder re-sort) for both embedders, with stage timings |
| 4 | The retrieved chunks written to `outputs/retrieved_chunks.md` |
| 5-6 | Retrieval metrics, rank shift per question, results by question type, cost of running the cross-encoder on every chunk |
| 7-8 | Qwen3.5-9B split over two GPUs; 100 answers (25 questions x 2 embedders x 2 pipelines) |
| 9 | All answers side by side |
| 10 | Answers graded by hand against gold key facts |
| 11 | Latency table, baseline vs re-ranking pipeline |
| 12 | Failure-mode analysis: the failure queries, the failure types the assignment lists, and Answer 1 vs Answer 2 in detail |
| 13-14 | Architecture comparison and the scaling question; checklist mapping every assignment item to a section |

## Models
| Role | Model |
|---|---|
| Bi-encoder | `BAAI/bge-small-en-v1.5` and `google/embeddinggemma-2` (float32) |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | `Qwen/Qwen3.5-9B`, fp16, thinking off, `device_map="auto"` over two 11 GB GPUs |

## Results (details in the notebook and in `outputs/`)
Retrieval over the 25 questions (gold chunk = the passage that answers the question):

| | bge-small | EmbeddingGemma 2 |
|---|---|---|
| Gold chunk in top 3, baseline | 18 | 16 |
| Gold chunk in top 3, re-ranked | 21 | 21 |
| Gold chunk anywhere in the 10 candidates | 22 | 23 |
| MRR, baseline -> re-ranked | 0.613 -> 0.711 | 0.645 -> 0.685 |

Hand-graded answers (25 per row): `bge-small` baseline 18 correct, 3 partly, 1 wrong rule, 3 abstain; re-ranked 19 / 4 / 0 / 2.
`embeddinggemma-2` baseline 15 / 6 / 0 / 4; re-ranked 17 / 5 / 0 / 3. The benefit is concentrated in the first 8 questions, which were picked
because retrieval struggles on them; on the other 17 ordinary questions re-ranking changed little.

The cross-encoder adds about 17 ms per question. Running it over all 1,504 chunks instead of 10 takes about 1.9 s.

## Files
* `rag_assignment.ipynb`: the notebook.
* `data/SRD_CC_v5.1.pdf`: the document used.
* `outputs/retrieved_chunks.md` and `retrieval.json`: the retrieved chunks, ranks and scores for every question and embedder.
* `outputs/generations_qwen35_9b.json`, `answers_graded.csv`, `verdicts_per_question.csv`: the 100 answers and their verdicts.
* `outputs/metrics.csv`, `per_question_ranks.csv`, `by_question_type.csv`, `failure_analysis.csv`, `latency.csv`, `cross_encoder_scaling.csv`: tables from the notebook.
* `outputs/thinking_attempt_partial.json`: an attempt with Qwen's thinking mode on, which ran out of tokens (15 of 16 tries) and was therefore not evaluated.

## Run it
```
pip install -r requirements.txt
# needs two GPUs of about 11 GB; the 9B model is about 18 GB in fp16
# optional environment variables: RAG_PDF (path of the PDF), RAG_OUT (output folder), MODEL_DIR (local copy of Qwen/Qwen3.5-9B)
jupyter nbconvert --to notebook --execute rag_assignment.ipynb
```
Tested with torch 2.11 (CUDA 12.8), transformers 5.20, sentence-transformers 6.1 on RTX 2080 Ti cards. EmbeddingGemma 2 also needs `pillow` and
`torchvision` because its processor is multimodal. On these cards use float16 for Qwen and float32 for EmbeddingGemma 2 (no native bfloat16).

## Limits
25 questions on one rulebook is small, and 8 of them were chosen because retrieval fails on them. The gold passages, gold key facts and answer
verdicts were written and judged by hand. The cross-encoder is a general web-search model, not trained on rules text.

## Attribution
This work includes material taken from the System Reference Document 5.1 ("SRD 5.1") by Wizards of the Coast LLC and available at
https://dnd.wizards.com/resources/systems-reference-document. The SRD 5.1 is licensed under the Creative Commons Attribution 4.0
International License available at https://creativecommons.org/licenses/by/4.0/legalcode.
