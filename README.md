<div align="center">
  <img src="assets/banner.svg" alt="Banner">
</div>

<p align="center">
<a href="https://github.com/ishandutta2007/Awesome-Awesome-Awesome"><img src="https://img.shields.io/badge/Awesome-%E2%9C%94-blueviolet?style=flat-square&logo=github" alt="Awesome"/></a>
<a href="https://discord.gg/jc4xtF58Ve"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord" /></a>
<a href="https://github.com/ishandutta2007"><img alt="GitHub followers" src="https://img.shields.io/github/followers/ishandutta2007?label=Follow" /></a>
</p>

# Awesome-Group-Query-Attention

## 🧠 Grouped-Query Attention: History, Progression, Variants, & Applications

**Grouped-Query Attention (GQA)** represents a foundational paradigm shift in the structural design and inference scaling of autoregressive Large Language Models (LLMs). Formally introduced by Ainslie et al. (Google Research) in May 2023 ("GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"), GQA established an optimal architectural compromise between multi-head attention and multi-query attention. 

Prior to GQA, LLM serving architecture faced a brutal trade-off: deploy Multi-Head Attention (MHA) for maximum conversational accuracy at the cost of crippling memory bottlenecks, or use Multi-Query Attention (MQA) to slash memory overhead while severely degrading text quality. GQA inverted this compromise, proving that grouping queries together to share a **bounded number of Key-Value (KV) heads** can deliver near-MHA quality alongside near-MQA speed, establishing an ideal balance for modern hardware accelerators.

---

## 📅 1. The Macro Chronological Evolution

The implementation of sequence model attention mechanisms has transitioned from memory-heavy individual projection states to aggressively pooled configurations, shifting toward modern variable-ratio clustering and dynamic page-allocated streaming frameworks.

```mermaid
graph LR
    MHA[Multi-Head Attention <br> Vaswani, 2017] --> MQA[Multi-Query Attention <br> Shazeer, 2019]
    MQA --> GQA[Grouped-Query Attention <br> Ainslie, 2023]
    GQA --> MLA[Multi-head Latent Attention <br> DeepSeek, 2024+]
    
    style MHA fill:#f9f,stroke:#333,stroke-width:1px
    style MQA fill:#bbf,stroke:#333,stroke-width:1px
    style GQA fill:#bfb,stroke:#333,stroke-width:1px
    style MLA fill:#fbb,stroke:#333,stroke-width:1px
```

| Architecture / Concept | Concept | Limitation / Significance | Year | Paper Link |
| :--- | :--- | :--- | :--- | :--- |
| **[The Unbounded Projection Era (Multi-Head Attention / MHA)](details/mha.md)** | The foundational attention mapping standard. Every individual Query head ($Q$) possesses its own distinct Key ($K$) and Value ($V$) head mapping projection space. | *Limitation:* Caused severe hardware memory capacity exhaustion during model serving. As sequence lengths scale out, storing the distinct Key-Value histories (KV Cache) for every single head consumes vast blocks of GPU VRAM, capping inference concurrency metrics. | 2017 | [Vaswani et al.](https://arxiv.org/abs/1706.03762) |
| **[The Aggressive Pooling Crash (Multi-Query Attention / MQA)](details/mqa.md)** | Drastically flattened the memory footprint. MQA collapsed the architectural projection layer down to a singular, solitary Key and Value head shared by all Query heads simultaneously. | *Limitation:* Created structural information capacity bottlenecks. Forcing dozens of complex, distinct Query expressions to route through a single KV channel degraded target model capacity, rendering large foundation architectures unstable or prone to semantic degradation. | 2019 | [Shazeer](https://arxiv.org/abs/1911.02150) |
| **[The Balanced Clustering Integration (Grouped-Query Attention / GQA)](details/gqa.md)** | Engineered a flexible midway architecture. GQA organizes Query heads into discrete, isolated groups where each separate partition shares a single local Key-Value head allocation block. | *Significance:* Delivered massive hardware deployment performance leaps. GQA allows models like Llama 3 to retain the expressive contextual indexing traits of MHA while scaling out real-world generation speeds to match MQA frameworks. | 2023 | [Ainslie et al.](https://arxiv.org/abs/2305.13245) |

---

## ⚙️ 2. Core Functional & Custom Configurations

Modern Transformer attention layers are structurally parameterized around the absolute proportion of total Query head groups mapped across underlying physical memory layouts.

| Configuration | Mechanism | Year | Paper Link |
| :--- | :--- | :--- | :--- |
| **[Mathematical Key-to-Query Mapping Ratios](details/mapping_ratios.md)** | Let $H_Q$ represent total Query heads and $H_{KV}$ represent Key-Value heads. The configuration controls the spatial compression layout: <br> $$G = \frac{H_Q}{H_{KV}}$$ <br> - When $H_{KV} = H_Q$ ($G=1$), the layout behaves identically to standard **Multi-Head Attention**. <br> - When $H_{KV} = 1$ ($G=H_Q$), the layout contracts into basic **Multi-Query Attention**. <br> - When $1 < H_{KV} < H_Q$, the system operates as **Grouped-Query Attention**, typically setting $G = 8$ for optimal balance. | 2023 | [Ainslie et al.](https://arxiv.org/abs/2305.13245) |
| **[Uptraining From Baseline MHA Checkpoints](details/uptraining.md)** | Converts an existing pre-trained MHA model into a GQA framework without needing a full pre-training run. It takes the $H_Q$ to $H_{KV}$ individual MHA keys, applies a spatial **Mean Pooling** dimension reduction across the distinct head matrices within each designated group, and runs a brief token fine-tuning phase to recalibrate projection weights. | 2023 | [Ainslie et al.](https://arxiv.org/abs/2305.13245) |

---

## 🏗️ 3. High-Capacity Architectural & Scaling Classes

Depending on token length requirements or extreme throughput objectives, attention pooling utilizes specialized mathematical modifications.

| Architecture Class | The Shift | Year | Paper Link |
| :--- | :--- | :--- | :--- |
| **[Dynamic Matrix Grouping (Variable Multi-Group Topology)](details/dynamic_matrix_grouping.md)** | Rather than freezing fixed head groups across every single layer block uniformly, variable topologies deploy broad MHA structures across early processing layers to capture fine-grained pixel or text roots, while switching to hyper-compressed GQA grouping setups deep inside internal hidden blocks. | 2023 | N/A |
| **[Low-Rank Compressional Mapping (Multi-head Latent Attention / MLA)](details/mla.md)** | Decouples attention mechanics further by compressing the KV Cache into a tiny, low-rank latent vector space. Instead of grouping separate heads explicitly, MLA mathematically compresses keys and values into a shared vector, expanding them back out instantly during execution to skip physical VRAM scaling walls entirely. | 2024 | [DeepSeek-AI](https://arxiv.org/abs/2405.04434) |

```mermaid
flowchart TB
    subgraph Group 1
        Q1[Q] --> KV1[K V]
        Q2[Q] --> KV1
        Q3[Q] --> KV1
        Q4[Q] --> KV1
    end
    subgraph Group 2
        Q5[Q] --> KV2[K V]
        Q6[Q] --> KV2
        Q7[Q] --> KV2
        Q8[Q] --> KV2
    end
    subgraph Group 3
        Q9[Q] --> KV3[K V]
        Q10[Q] --> KV3
        Q11[Q] --> KV3
        Q12[Q] --> KV3
    end
    subgraph Group 4
        Q13[Q] --> KV4[K V]
        Q14[Q] --> KV4
        Q15[Q] --> KV4
        Q16[Q] --> KV4
    end
```

---

## 🛠️ 4. Production Engineering Challenges & Hardware Solutions

Deploying dense GQA frameworks across enterprise inferencing infrastructure introduces compilation quirks and memory alignment considerations.

| Challenge | The Problem | Mitigation | Year | Paper Link |
| :--- | :--- | :--- | :--- | :--- |
| **[The Non-Uniform Tensor Layout Compilation Penalty](details/tensor_layout_penalty.md)** | Traditional tensor layout kernels are heavily optimized for balanced, perfectly symmetric matrix computations. If the group configuration $G$ does not cleanly align with the physical hardware thread configurations or SIMD width of the target GPU, memory bandwidth efficiency plummets. | Implementing specialized **FlashAttention** kernels explicitly compiled for GQA architectures. These custom setups bypass intermediate GPU global memory reads by executing the grouped head matrix operations entirely inside local SRAM registers. | 2022 | [Dao et al. (FlashAttention)](https://arxiv.org/abs/2205.14135) |
| **[The KV Cache Memory Striding Mismatch](details/memory_striding.md)** | During long-context batch serving, pointer layouts for pooled KV channels can break typical linear memory reading operations, causing latency spikes when fetching token records from scattered memory blocks. | Pairing GQA directly with **PagedAttention** architectures. PagedAttention fragments the grouped KV cache into structured, uniform virtual memory pages, ensuring clean hardware execution paths regardless of head counts. | 2023 | [Kwon et al. (PagedAttention)](https://arxiv.org/abs/2309.06180) |

---

## 🚀 5. Frontier Real-World AI Infrastructure Applications

| Application | Details | Year | Paper Link |
| :--- | :--- | :--- | :--- |
| **[High-Concurrency Enterprise LLM Serving (vLLM / TensorRT-LLM)](details/enterprise_serving.md)** | Maximizes simultaneous user request throughput. Transitioning open architectures to GQA configurations allows cloud hosting platforms to multiply batch capacities per GPU card without encountering out-of-memory errors. | 2023 | [vLLM Paper](https://arxiv.org/abs/2309.06180) |
| **[Massive Long-Context Sequence Parsing (Gemini / Llama 3 Ultra-Long)](details/long_context.md)** | Unlocks massive text processing horizons (up to 128K+ tokens). GQA prevents the model's tracking cache from inflating exponentially, allowing systems to ingest complete document libraries or codebases in a single prompt. | 2023 | N/A |
| **[Edge-Device On-Chip Deep Learning Executions (Apple Silicon / Snapdragon)](details/edge_device.md)** | Minimizes processing overhead on localized consumer devices. Compressing model attention structures via GQA lowers memory footprints enough to serve interactive assistants directly on local mobile hardware and laptops. | 2023 | N/A |

---

## 📚 References
1. Vaswani, A., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS)*.
2. Shazeer, N. (2019). Fast transformer decoding: One write-head is all you need. *arXiv preprint arXiv:1911.02150*.
3. Ainslie, J., et al. (2023). GQA: Training generalized multi-query transformer models from multi-head checkpoints. *arXiv preprint arXiv:2305.13245*.

---

To advance this documentation repository, scaling architecture, or MLOps automation pipeline, consider exploring these adjacent development pathways:

* Build a **Python script using PyTorch** demonstrating how to expand a grouped GQA Key-Value tensor up to a matching Query tensor shape using the `torch.repeat_interleave` function.
* Generate a **comprehensive Markdown table** explicitly comparing Multi-Head Attention (MHA), Multi-Query Attention (MQA), Grouped-Query Attention (GQA), and Multi-head Latent Attention (MLA) across VRAM cache scaling rates, training stability metrics, and serving latencies.

***

<!-- SEO: Awesome Group Query Attention, LLM, Large Language Models, Generative AI, Transformer, Multi-Head Attention, Multi-Query Attention -->

##  Star History
<div align="center">
<a href="https://www.star-history.com/?repos=ishandutta2007%2FAwesome-Group-Query-Attention&type=date&legend=bottom-right">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=ishandutta2007/Awesome-Group-Query-Attention&type=date&theme=dark&legend=bottom-right" />
<source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=ishandutta2007/Awesome-Group-Query-Attention&type=date&legend=bottom-right" />
<img alt="Star History Chart" src="https://api.star-history.com/chart?repos=ishandutta2007/Awesome-Group-Query-Attention&type=date&legend=bottom-right" />
</picture>
</a>
</div>
