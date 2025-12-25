# VLM & AI Safety Engineering Study Path

This guide is designed to move you from "vibe coding" to **Principled Engineering** in the domain of Vision-Language Models (VLMs) and Safety Alignment. It connects the theoretical concepts directly to the code we've been working on (LOSF, LEACE, ETA).

---

## Phase 1: The Architecture (How the Brain Works)

### 1. The Transformer & Attention (The Foundation)
*   **Concept:** Everything in modern AI (LLMs, ViTs) is based on the **Transformer**.
*   **Key Mechanism:** **Self-Attention**. It allows the model to look at all parts of the input at once and decide what's relevant to what.
    *   *Analogy:* When reading the sentence "The animal didn't cross the street because it was too tired," attention helps the model understand that "it" refers to "animal," not "street."
*   **Study Checklist:**
    *   [ ] Read **"The Illustrated Transformer"** by Jay Alammar (Essential visual guide).
    *   [ ] Understand **Query, Key, Value ($Q, K, V$)** matrices.
    *   [ ] **Code Link:** In our code, `model.layers` are stacks of these transformer blocks.

### 2. Vision Transformers (ViT) (The Eyes)
*   **Concept:** How do we feed an image to a Transformer? We chop it into patches (e.g., 16x16 pixels) and treat each patch like a word token.
*   **Relevance:** The "Vision Encoder" in our project (e.g., CLIP-ViT-L/14) is exactly this. It turns an image into a sequence of vectors ($Z$).
*   **Study Checklist:**
    *   [ ] **"An Image is Worth 16x16 Words"** (The ViT Paper).
    *   [ ] Concept: **Patch Embeddings** & **Positional Encodings**.

### 3. CLIP (The Bridge)
*   **Concept:** CLIP trains a Vision Encoder and a Text Encoder to output vectors in the *same mathematical space*.
*   **Mechanism:** It pulls the vector for an image of a dog close to the vector for the text "A photo of a dog."
*   **Relevance:** Our safety metrics (CLIPScore) and alignment methods rely on this shared space. If "Hate" is a direction in text space, it's likely a similar direction in image space.

---

## Phase 2: The Mathematics of Representations (The Logic of LOSF)

This is the specific math used in **Visionencoder_Rep_Energyfilter (LOSF)** and **LEACE**.

### 1. The Linear Representation Hypothesis
*   **Theory:** Complex concepts (like "Gender," "Sentiment," or "Hatefulness") are represented as **linear directions** (vectors) in the model's activation space.
*   **Why it matters:** If "Hate" is just an arrow pointing North-East in the high-dimensional space, we can "erase" it by flattening the data along that specific arrow.

### 2. Covariance & PCA (Finding the Arrow)
*   **Covariance Matrix ($\Sigma$):** A map of how data varies. In our project, we calculated $\Sigma_{Harmful}$.
*   **Eigenvalues ($\lambda$) & Eigenvectors ($U$):**
    *   **Eigenvector ($U$):** The *direction* of maximum variance (The "Hate" Arrow).
    *   **Eigenvalue ($\lambda$):** The *strength* (Energy) of that variance.
*   **In Our Code:** We looked for directions with huge $\lambda$ (high energy) in the harmful dataset and shrank them.

### 3. Whitening (ZCA)
*   **Concept:** Transforming the data so the "Safe" concepts look like a perfect sphere (Standard Normal Distribution).
*   **Why:** It makes it easier to spot outliers. If the safe data is a sphere, the harmful data sticks out like a spike.
*   **Math:** $Z_{whitened} = Z \cdot W^T$.

---

## Phase 3: VLM Architectures (The Body)

### 1. LLaVA / InstructBLIP
*   **Structure:** `[Vision Encoder] --> [Projector] --> [LLM]`
*   **The Projector:** A small neural network (often a Linear Layer or MLP) that translates "Image Vector Language" into "Text Token Language."
*   **Inference:** The LLM sees the image features as just another prompt prefix.

---

## Phase 4: Practical Engineering Skills

### 1. PyTorch Tensor Manipulation
*   **Broadcasting:** Adding a `(1, D)` vector to a `(B, T, D)` tensor.
*   **Einsum:** The "Swiss Army Knife" of matrix multiplication.
    *   *Example from our code:* `torch.einsum('btd,dk->btk', Z, W.T)` means "Multiply input $Z$ by matrix $W$."

### 2. HuggingFace Transformers Library
*   **AutoModel / AutoTokenizer:** The standard API for loading models.
*   **Hooks (`register_forward_hook`):** The magic tool we used to intercept the vision encoder inside the massive LLaVA model without rewriting the whole library.

---

## Recommended "Zero-to-Hero" Reading List

1.  **Blog:** [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) - *Start Here.*
2.  **Blog:** [The Illustrated GPT-2 (Jay Alammar)](https://jalammar.github.io/illustrated-gpt2/) - *Visualizing activations.*
3.  **Paper:** [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) - *The foundation of modern VLMs.*
4.  **Paper:** [LLaVA: Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) - *How to connect Vision to LLMs.*
5.  **Advanced:** [LEACE: Concept Erasure by Linear Guardrails](https://arxiv.org/abs/2306.03819) - *The math behind our safety method.*
