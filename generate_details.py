import os
import re

readme_path = "README.md"
with open(readme_path, "r", encoding="utf-8") as f:
    content = f.read()

os.makedirs("details", exist_ok=True)

# Define the 12 items and their details
items = [
    ("The Unbounded Projection Era (Multi-Head Attention / MHA)", "mha", "Multi-Head Attention", "graph TD; Q1-->K1; Q1-->V1; Q2-->K2; Q2-->V2;"),
    ("The Aggressive Pooling Crash (Multi-Query Attention / MQA)", "mqa", "Multi-Query Attention", "graph TD; Q1-->K1; Q2-->K1; Q3-->K1; Q1-->V1; Q2-->V1; Q3-->V1;"),
    ("The Balanced Clustering Integration (Grouped-Query Attention / GQA)", "gqa", "Grouped-Query Attention", "graph TD; Q1-->K1; Q2-->K1; Q3-->K2; Q4-->K2;"),
    ("Mathematical Key-to-Query Mapping Ratios", "mapping_ratios", "Mathematical Key-to-Query Mapping Ratios", "graph LR; HQ-->Ratio; HKV-->Ratio; Ratio-->GQA;"),
    ("Uptraining From Baseline MHA Checkpoints", "uptraining", "Uptraining From Baseline MHA Checkpoints", "graph TD; MHA-->MeanPooling; MeanPooling-->GQA_Init; GQA_Init-->FineTuning;"),
    ("Dynamic Matrix Grouping (Variable Multi-Group Topology)", "dynamic_matrix_grouping", "Dynamic Matrix Grouping", "graph TD; EarlyLayers-->MHA; DeepLayers-->GQA;"),
    ("Low-Rank Compressional Mapping (Multi-head Latent Attention / MLA)", "mla", "Multi-head Latent Attention", "graph TD; KV_Cache-->Compression; Compression-->LatentVector; LatentVector-->Expansion;"),
    ("The Non-Uniform Tensor Layout Compilation Penalty", "tensor_layout_penalty", "The Non-Uniform Tensor Layout Compilation Penalty", "graph TD; GQA-->FlashAttention; FlashAttention-->SRAM;"),
    ("The KV Cache Memory Striding Mismatch", "memory_striding", "The KV Cache Memory Striding Mismatch", "graph TD; GQA-->PagedAttention; PagedAttention-->UniformPages;"),
    ("High-Concurrency Enterprise LLM Serving (vLLM / TensorRT-LLM)", "enterprise_serving", "Enterprise LLM Serving", "graph TD; UserRequests-->Batching; Batching-->vLLM; vLLM-->GQA_Optimization;"),
    ("Massive Long-Context Sequence Parsing (Gemini / Llama 3 Ultra-Long)", "long_context", "Massive Long-Context Sequence Parsing", "graph TD; MassiveText-->GQA; GQA-->BoundedCache;"),
    ("Edge-Device On-Chip Deep Learning Executions (Apple Silicon / Snapdragon)", "edge_device", "Edge-Device On-Chip Deep Learning Executions", "graph TD; Mobile-->GQA; GQA-->LowMemoryFootprint;")
]

for title, filename, heading, diagram in items:
    md_content = f"# {heading}\n\n## Overview\nDetailed information about {heading} in the context of Grouped-Query Attention.\n\n## Diagram\n```mermaid\n{diagram}\n```\n\n[Back to README](../README.md)"
    with open(f"details/{filename}.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    
    # Replace the text in README to include the link
    # Using regex to find the exact bolded text and add link if not present
    safe_title = title.replace('(', r'\(').replace(')', r'\)')
    pattern = r'(\*\*' + safe_title + r'\*\*)'
    replacement = rf'**[{title}](details/{filename}.md)**'
    content = re.sub(pattern, replacement, content, count=1)

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(content)
