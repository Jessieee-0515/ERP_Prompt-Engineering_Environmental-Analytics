# Appendix RAG (P6) Configuration

This appendix documents the full configuration of the P6 (RAG) prompt condition described in Section 3.4.6, including the knowledge base contents, the segmentation methodology, the retrieval configuration, and the core retrieval code, to support reproducibility and review.

## 1. Knowledge Base

The knowledge base consists of 16 retrieval chunks drawn from five independent sources plus one general physical-background source: Piccolroaz et al. (2013, HESS), Toffolon et al. (2014, Limnology and Oceanography), Piccolroaz (2016, Advances in Oceanography and Limnology), Piotrowski et al. (2022, Limnologica), a synthesis of air2water model-family literature, and a general lake/ocean surface energy-balance framework.

### Table 1. Knowledge Base Chunk Inventory

| Chunk ID | Source | Layer | Content Summary |
|---|---|---|---|
| CHUNK_1 | Piccolroaz et al. 2013 (HESS) | Core | Model positioning: air2water is a semi-physical lumped-parameter model; the full version has 8 parameters, with 6- and 4-parameter simplified versions also proposed. |
| CHUNK_1b | General lake/ocean surface energy-balance framework | Core | Derivation of the governing ODE from first-principles energy conservation (Qnet = Qsw − Qlw − Qsensible − Qlatent → ρ·cp·V·dTw/dt = Qnet·A). |
| CHUNK_2 | Piccolroaz et al. 2013 (HESS) | Core | Governing equation body: the dTw/dt expression and the piecewise δ (normalized mixed-layer thickness) function. |
| CHUNK_3 | Piccolroaz et al. 2013 (HESS) | Core | Definition table for the 8 parameters (p1–p8): physical meaning and units. |
| CHUNK_3b | Piccolroaz et al. 2013 (HESS) | Extended | Heat-flux decomposition (Eq. 1): correspondence between coefficients c1–c5 and parameters p1–p5. |
| CHUNK_3c | Piccolroaz et al. 2013 (HESS) | Extended | Physically plausible search ranges for the 8 parameters, usable as optimizer bounds. |
| CHUNK_3d | Piccolroaz et al. 2013 (HESS) | Extended | Rationale for the 8→6→4 parameter simplification hierarchy. |
| CHUNK_3e | Piccolroaz et al. 2013 (HESS) | Extended | Original GLUE calibration method and the Nash–Sutcliffe efficiency (NSE) definition. |
| CHUNK_3f | Toffolon et al. 2014 (Limnology and Oceanography) | Core | Cross-lake validation across 14 lakes (NSE > 0.87); parameter–mean-depth regression relationship. |
| CHUNK_3g | Adapted from this study's literature review | Extended | Motivation for including a RAG condition in the prompt-strategy comparison. |
| CHUNK_3h | Piccolroaz (2016), Advances in Oceanography and Limnology | Extended | Effect of missing-data rate and calibration-window length on overfitting risk. |
| CHUNK_3i | Piotrowski et al. 2023 (Limnologica) | Extended | Nine-parameter variant; comparison of 8 optimization algorithms for model calibration. |
| CHUNK_3j | Synthesis of air2water model-family literature | Extended | Overview of 6/9-parameter variants, the ice-cover extension, and the air2stream river variant. |
| CHUNK_4 | Piccolroaz et al. 2013 (HESS) | Core | Numerical solution method: explicit Euler scheme with a one-day time step. |
| CHUNK_5 | Piccolroaz et al. 2013 (HESS) | Core | Study area (Lake Superior) and NDBC data background. |
| CHUNK_6 | Usage-constraint note | Core | Instruction against hard-coding example parameter values from a different calibration window. |

## 2 Knowledge Segmentation

Each chunk was defined as a single self-contained conceptual or physical unit (e.g., one equation block, one parameter table, one methodological claim) so that retrieval operates over semantically coherent units.

Chunks were further assigned to one of two layers. The **core layer** (8 chunks: CHUNK_1, 1b, 2, 3, 3f, 4, 5, 6) contains information indispensable for producing a structurally correct implementation — the governing equation, parameter definitions, the numerical solver specification, the defining case study, and usage constraints. During pilot testing (Section 3.4.5), a pure top-k similarity search over the full 16-chunk pool was found to occasionally omit the governing equation and parameter-table chunks, because these chunks are symbol-dense and text-sparse and therefore score lower on embedding-based semantic similarity than more narrative chunks. To prevent this failure mode, core-layer chunks are exempted from similarity ranking and are always injected in full.

The **extended layer** (8 chunks: CHUNK_3b, 3c, 3d, 3e, 3g, 3h, 3i, 3j) contains auxiliary or contextual knowledge — cross-paper corroboration, methodological justification, and model-family disambiguation — that supports but is not strictly required for a correct implementation. These chunks are ranked by cosine similarity to the task query, and only the most relevant subset is retrieved, which keeps the injected context bounded in length (see Section 3).

## 3 Retrieval Configuration
 proceeds in five stages. Each chunk is embedded once and cached, core-layer chunks are always retained, and extended-layer chunks are ranked by cosine similarity to the task query and filtered to the top four before being merged into the final P6 prompt. Table 2 below specifies the exact parameters governing each stage of this pipeline.

### Table 2. Configuration of RAG

| Configuration item | Value / Description |
|---|---|
| Embedding model | OpenAI text-embedding-3-small |
| Similarity metric | Cosine similarity |
| Retrieval strategy | Hybrid: core-layer chunks mandatory; extended-layer chunks ranked by similarity |
| Core-layer size | 8 chunks (CHUNK_1, 1b, 2, 3, 3f, 4, 5, 6) — excluded from similarity ranking, always injected |
| Extended-layer pool / Top-k | 8 candidate chunks; top 4 selected by cosine similarity to the task query |
| Embedding cache | Keyed by a 16-character SHA-256 hash of the chunk text; cache entries are invalidated automatically if the underlying text changes |
