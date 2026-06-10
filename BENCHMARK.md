# Encoder Retrieval Benchmark (end-to-end, real corpus)

- **Corpus:** 2995 real arXiv papers (title+abstract), built from 150 distractors/query + gold papers.
- **Gold present in corpus:** 54/54
- **Index:** Faiss `IndexFlatIP` (exact cosine); ANN cross-check via `IndexHNSWFlat`.
- **Baseline:** local `all-MiniLM-L6-v2` (never an API). **Variable changed:** the encoder only.

| Metric | base | fine-tuned | delta |
|--------|------|-----------|-------|
| Recall@10 | 0.2500 | 0.3264 | +0.0764 |
| MRR | 0.1404 | 0.2543 | +0.1139 |
| NDCG@10 | 0.1357 | 0.2260 | +0.0903 |
| NLI relevance | 0.1580 | 0.1249 | -0.0331 |

Faiss `IndexHNSWFlat` ANN recall@10 vs exact (fine-tuned encoder): **1.0000**

## Per-query Recall (fine-tuned)

| Query | Recall | MRR | NDCG |
|-------|--------|-----|------|
| post-training quantization for large language models | 0.00 | 0.00 | 0.00 |
| policy gradient methods for deep reinforcement learning | 0.33 | 0.12 | 0.15 |
| denoising diffusion probabilistic models for image generation | 1.00 | 0.50 | 0.71 |
| transformer self-attention architecture for sequence modeling | 0.50 | 0.10 | 0.18 |
| vision transformers for image classification | 0.00 | 0.00 | 0.00 |
| parameter-efficient fine-tuning of pretrained language models | 0.00 | 0.00 | 0.00 |
| retrieval-augmented generation for knowledge-intensive question answering | 0.00 | 0.00 | 0.00 |
| contrastive learning for self-supervised visual representations | 0.33 | 1.00 | 0.47 |
| sparsely gated mixture of experts for scaling neural networks | 0.67 | 0.12 | 0.28 |
| graph neural networks for node classification | 0.00 | 0.00 | 0.00 |
| generative adversarial networks for image synthesis | 0.00 | 0.00 | 0.00 |
| real-time object detection in images | 0.33 | 0.50 | 0.30 |
| knowledge distillation for neural network compression | 0.00 | 0.00 | 0.00 |
| deep residual and densely connected convolutional networks | 0.67 | 1.00 | 0.67 |
| distributed word representations from large text corpora | 0.50 | 0.11 | 0.18 |
| batch normalization for training deep neural networks | 1.00 | 1.00 | 1.00 |
| adaptive gradient optimization for stochastic training | 0.00 | 0.00 | 0.00 |
| variational autoencoders for generative modeling | 0.00 | 0.00 | 0.00 |
| adversarial examples and robustness of neural networks | 0.50 | 0.33 | 0.31 |
| neural style transfer for artistic image generation | 0.00 | 0.00 | 0.00 |
| sequence to sequence learning with neural networks | 0.50 | 1.00 | 0.61 |
| scaling laws for large language models | 0.50 | 0.14 | 0.20 |
| network pruning for efficient deep learning | 0.00 | 0.00 | 0.00 |
| dynamic routing between capsules | 1.00 | 0.17 | 0.36 |
