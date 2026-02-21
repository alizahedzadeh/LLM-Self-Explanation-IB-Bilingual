# The Sufficiency-Conciseness Trade-off in LLM Self-Explanation

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/[your-arxiv-number])
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)

Research code for the paper: **"The Sufficiency-Conciseness Trade-off in LLM Self-Explanation from an Information Bottleneck Perspective"**

## 📄 Abstract

Large Language Models increasingly rely on self-explanations, such as chain of thought reasoning, to improve performance on multi step question answering. While these explanations enhance accuracy, they are often verbose and costly to generate, raising the question of how much explanation is truly necessary. In this paper, we examine the trade-off between sufficiency, defined as the ability of an explanation to justify the correct answer, and conciseness, defined as the reduction in explanation length. Building on the information bottleneck principle, we conceptualize explanations as compressed representations that retain only the information essential for producing correct answers.To operationalize this view, we introduce an evaluation pipeline that constrains explanation length and assesses sufficiency using multiple language models on the ARC Challenge dataset. To broaden the scope, we conduct experiments in both English, using the original dataset, and Persian, as a resource-limited language through translation. Our experiments show that more concise explanations often remain sufficient, preserving accuracy while substantially reducing explanation length, whereas excessive compression leads to performance degradation.

## 🎯 Key Questions

- **How sufficient are LLM self-explanations?** We measure accuracy vs. explanation length trade-offs
- **Do explanations transfer across languages?** Cross-lingual analysis (English/Persian)
- **Which models have better explanation quality?** Comparative study of 7 LLMs

## 🤖 Models Tested

- GPT-4o Mini, Claude 3 Haiku, Gemini 2.0 Flash
- Llama 4 Scout
- DeepSeek V3, Mistral Large, Cohere Command R

## 📊 Dataset

- **Source**: ARC Dataset (Challenge) [https://www.kaggle.com/datasets/jeromeblanchet/arc-ai2-reasoning-challenge]
- **Size**: 2,581 questions
- **Languages**: English + Persian (translated)
- **Constraint Levels**: Z0 (full) → Z10-Z90 (10%-90% retained)


## 📖 Citation

```bibtex
@misc{zahedzadeh2026sufficiencyconcisenesstradeoffllmselfexplanation,
      title={The Sufficiency-Conciseness Trade-off in LLM Self-Explanation from an Information Bottleneck Perspective}, 
      author={Ali Zahedzadeh and Behnam Bahrak},
      year={2026},
      eprint={2602.14002},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.14002}, 
}
```

## 📧 Contact

Questions?

Open an issue or contact: [alizahedzadeh7@gmail.com]

---

**Paper**: [https://arxiv.org/abs/2602.14002](https://arxiv.org/abs/XXXX.XXXXX)