# content_data.py
# Curated Pedagogical Theory, Mathematical Formulations, Hands-On Tasks, Graded Quizzes & Reference Platforms
# Developed by Dr. Rohan Borgalli, Shah & Anchor Kutchhi Engineering College, Chembur

REFERENCE_PLATFORMS = [
    {
        "name": "TensorFlow Neural Network Playground",
        "category": "ML & Deep Learning",
        "url": "https://playground.tensorflow.org/",
        "desc": "Interactive browser simulation of neural network layers, activations, learning rates, and non-linear decision boundaries."
    },
    {
        "name": "Machine Learning Playground",
        "category": "Classical ML Classifiers",
        "url": "https://ml-playground.com/",
        "desc": "Visual experimentation with SVM, KNN, Decision Trees, and MLP hyperparameters on synthetic 2D datasets."
    },
    {
        "name": "Seeing Theory (Brown University)",
        "category": "Probability & Statistics",
        "url": "https://seeing-theory.brown.edu/",
        "desc": "Visual intuition for probability distributions, Bayesian inference, and ordinary least squares regression."
    },
    {
        "name": "SPIT Virtual Lab for AI & Deep Learning",
        "category": "Academic AI/DL Virtual Lab",
        "url": "https://vlab.spit.ac.in/",
        "desc": "Curriculum-aligned engineering virtual laboratory with step-by-step pre-tests, simulations, and post-tests."
    },
    {
        "name": "Google TensorFlow Embedding Projector",
        "category": "Embeddings & Dimensionality Reduction",
        "url": "https://projector.tensorflow.org/",
        "desc": "Interactive 2D/3D exploration of high-dimensional word and sentence embeddings via PCA, t-SNE, and UMAP."
    },
    {
        "name": "Cohere Embed & Rerank Playground",
        "category": "Semantic Search & Ranking",
        "url": "https://dashboard.cohere.com/",
        "desc": "Visual comparison of dense semantic embeddings, cosine similarity clustering, and cross-encoder reranking."
    },
    {
        "name": "Dify.ai / Pinecone RAG Workbench",
        "category": "RAG & Agentic Workflows",
        "url": "https://dify.ai/",
        "desc": "Visual pipeline orchestration showing document chunking, vector database retrieval, and ReAct agent tool calling."
    }
]

MODULES = {
    "mod1_ml_dl": {
        "id": "mod1_ml_dl",
        "title": "1. AI, Machine Learning & Deep Learning",
        "subtitle": "Supervised Learning, Multi-Layer Perceptrons (MLP), Backpropagation & Decision Boundaries",
        "theory": r"""
### 1.1 Foundational Hierarchy: AI vs. ML vs. Deep Learning
* **Artificial Intelligence (AI):** Any computational system capable of performing tasks that typically require human intelligence (symbolic reasoning, search, planning, perception).
* **Machine Learning (ML):** A subset of AI where algorithms learn parameterized patterns $f_\theta(x) \rightarrow \hat{y}$ directly from data without explicit rule programming.
* **Deep Learning (DL):** Stacked multi-layer neural networks that automatically learn hierarchical feature representations through differentiable composition.

### 1.2 Forward Pass & Non-Linear Activation Functions
Given an input feature vector $x \in \mathbb{R}^d$, each hidden layer $l$ computes an affine transformation followed by a non-linear activation $\sigma(\cdot)$:
$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \quad a^{(l)} = \sigma(z^{(l)})$$
* **ReLU (Rectified Linear Unit):** $\text{ReLU}(z) = \max(0, z)$ — prevents vanishing gradients in deep networks.
* **Hyperbolic Tangent (Tanh):** $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \in (-1, 1)$ — zero-centered smooth boundary.
* **Logistic Sigmoid:** $\sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)$ — maps logits to binary probabilities.

### 1.3 Cross-Entropy Loss & Gradient Descent Optimization
For binary classification with true label $y \in \{0, 1\}$ and predicted probability $\hat{y} \in (0, 1)$, the **Binary Cross-Entropy Loss** with $L_2$ regularization ($\alpha$) is:
$$\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^N \Big[ y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i) \Big] + \frac{\alpha}{2}\|W\|_2^2$$
Weights are iteratively updated via **Backpropagation** and **Gradient Descent**:
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$
where $\eta$ is the **Learning Rate**.
        """,
        "task_prompt": "Load either the preloaded 'EV Battery Diagnostics CSV', 'Synthetic Moons', or upload your own CSV. Tune the Learning Rate, Activation Function, and Hidden Layer Neurons until Training Accuracy exceeds 88%, then click 'Verify Sandbox Task'.",
        "quiz": [
            {
                "q": "What happens if the Learning Rate (η) in Gradient Descent is set excessively high?",
                "options": [
                    "The model converges instantaneously with zero error",
                    "The optimizer overshoots the loss minimum, causing oscillations or divergence",
                    "The neural network automatically adds more hidden layers",
                    "Overfitting is completely eliminated"
                ],
                "ans": 1
            },
            {
                "q": "Why are non-linear activation functions (such as ReLU or Tanh) required in hidden layers?",
                "options": [
                    "To compress CSV file sizes on disk",
                    "Because without non-linearity, any stack of linear layers mathematically collapses into a single linear transformation",
                    "To convert numerical features into strings",
                    "To bypass backpropagation"
                ],
                "ans": 1
            },
            {
                "q": "What is the role of L2 Regularization (α) during neural network training?",
                "options": [
                    "It penalizes overly large weights to prevent overfitting and produce smoother decision boundaries",
                    "It increases the number of input features",
                    "It forces the learning rate to zero",
                    "It deletes misclassified training points"
                ],
                "ans": 0
            }
        ]
    },
    "mod2_nlp_embed": {
        "id": "mod2_nlp_embed",
        "title": "2. NLP & Dense Vector Embeddings",
        "subtitle": "Tokenization, Sparse TF-IDF Matrices vs. 384-D Dense Bi-Encoder Embeddings & PCA Projection",
        "theory": r"""
### 2.1 Classical Lexical NLP: Tokenization & TF-IDF
Traditional Natural Language Processing converts text into sparse high-dimensional vectors of size $|V|$ (vocabulary length). **TF-IDF** weights a term $t$ in document $d$ across a corpus of $N$ documents:
$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{N}{1 + \text{DF}(t)}\right)$$
**Limitation:** Sparse lexical vectors have zero semantic awareness—words like *"Electric Vehicle"* and *"Battery Automobile"* share zero dot-product overlap if they use different surface tokens.

### 2.2 Dense Transformer Embeddings (`all-MiniLM-L6-v2`)
Modern bi-encoder Transformer models pass tokens through multi-head self-attention layers and apply **Mean Pooling** + **$L_2$ Normalization** to map any sentence into a compact dense vector $v \in \mathbb{R}^{384}$ with $\|v\|_2 = 1$:
$$v = \frac{\text{MeanPool}(H_{\text{last}})}{\|\text{MeanPool}(H_{\text{last}})\|_2}$$
Sentences with similar semantic meaning cluster close together in $\mathbb{R}^{384}$, which we project into 2D or 3D visual space via **Principal Component Analysis (PCA)** by computing the top eigenvectors of the covariance matrix $\Sigma = \frac{1}{M}X^T X$.
        """,
        "task_prompt": "Load the Multi-Domain Corpus (or type 6+ custom sentences across EVs, Deep Learning, and Indian Food), inspect the 384-D vector matrix vs. TF-IDF matrix, and toggle between the 2D and 3D PCA Embedding plots.",
        "quiz": [
            {
                "q": "Why do dense Transformer embeddings outperform Bag-of-Words and TF-IDF for semantic search?",
                "options": [
                    "Dense embeddings map synonyms and conceptually related phrases to nearby points in vector space even when exact words differ",
                    "Dense embeddings only store punctuation marks",
                    "TF-IDF requires a GPU cluster whereas Transformers do not",
                    "Dense embeddings have zero dimensions"
                ],
                "ans": 0
            },
            {
                "q": "What is the exact vector dimensionality produced by the `all-MiniLM-L6-v2` sentence embedding model?",
                "options": [
                    "16 dimensions",
                    "384 dimensions",
                    "15,000 dimensions",
                    "2 dimensions"
                ],
                "ans": 1
            },
            {
                "q": "What is the purpose of applying Principal Component Analysis (PCA) in the Embedding Sandbox?",
                "options": [
                    "To project 384-dimensional vectors onto the top 2 or 3 orthogonal axes of maximum variance so humans can visualize semantic clusters",
                    "To translate English text into Marathi",
                    "To encrypt the database",
                    "To increase token count"
                ],
                "ans": 0
            }
        ]
    },
    "mod3_vectordb": {
        "id": "mod3_vectordb",
        "title": "3. Vector Database Mechanics",
        "subtitle": "Cosine Similarity, Dot Product, Euclidean (L2) Distance & HNSW Indexing",
        "theory": r"""
### 3.1 How Vector Databases Store & Retrieve Knowledge
A **Vector Database** (ChromaDB, FAISS, Qdrant, Pinecone) stores dense vector embeddings $\{d_1, d_2, \dots, d_M\} \subset \mathbb{R}^n$ alongside document payloads and metadata. When a user submits a query, the query is embedded into $q \in \mathbb{R}^n$ and compared against stored vectors.

### 3.2 Core Distance & Similarity Metrics
1. **Cosine Similarity (Angle-Based, Scale-Invariant):**
   $$\text{CosSim}(q, d) = \frac{q \cdot d}{\|q\|_2 \|d\|_2} = \frac{\sum_{i=1}^n q_i d_i}{\sqrt{\sum_{i=1}^n q_i^2}\sqrt{\sum_{i=1}^n d_i^2}} \in [-1, 1]$$
2. **Inner / Dot Product:**
   $$\text{Dot}(q, d) = \sum_{i=1}^n q_i d_i \quad (\text{Identical to Cosine Similarity when } \|q\|_2 = \|d\|_2 = 1)$$
3. **Euclidean ($L_2$) Distance:**
   $$d_{L_2}(q, d) = \|q - d\|_2 = \sqrt{\sum_{i=1}^n (q_i - d_i)^2}$$
   For unit-normalized vectors, squared $L_2$ distance is monotonically related to Cosine Similarity via $\|q - d\|_2^2 = 2 - 2\,\text{CosSim}(q, d)$.
        """,
        "task_prompt": "Enter a custom technical query in the Vector DB Sandbox, inspect the step-by-step Query Vector vs. Document Vector calculation table, and compare the pairwise Cosine Similarity Matrix heatmap.",
        "quiz": [
            {
                "q": "When all embedding vectors are L2-normalized to unit length (||v|| = 1), Cosine Similarity is mathematically equal to:",
                "options": [
                    "The Dot Product (Inner Product)",
                    "The Manhattan (L1) Distance",
                    "Zero",
                    "The logarithm of document length"
                ],
                "ans": 0
            },
            {
                "q": "When ranking documents by Euclidean (L2) Distance vs. Cosine Similarity, which direction indicates the closest semantic match?",
                "options": [
                    "Higher Cosine Similarity (closer to +1.0) and Lower Euclidean L2 Distance (closer to 0.0)",
                    "Lower Cosine Similarity (-1.0) and Higher Euclidean L2 Distance",
                    "Both metrics should be exactly 100",
                    "Neither metric measures semantic closeness"
                ],
                "ans": 0
            },
            {
                "q": "Which graph-based indexing structure is used by ChromaDB for fast Approximate Nearest Neighbor (ANN) search?",
                "options": [
                    "HNSW (Hierarchical Navigable Small World)",
                    "Bubble Sort Linked List",
                    "FIFO Queue",
                    "relational SQL JOIN"
                ],
                "ans": 0
            }
        ]
    },
    "mod4_llm_prompt": {
        "id": "mod4_llm_prompt",
        "title": "4. LLMs & Prompt Engineering",
        "subtitle": "Autoregressive Next-Token Probabilities, Temperature, Top-P, Zero-Shot, Few-Shot & Chain-of-Thought",
        "theory": r"""
### 4.1 Autoregressive Decoding & Logit Scaling
Large Language Models (LLMs) are decoder-only Transformers that model the conditional probability of the next token $w_t$ given context sequence $w_{<t}$:
$$P(w_t = i \mid w_{<t}) = \text{Softmax}\left(\frac{z_t}{T}\right)_i = \frac{\exp(z_{t,i} / T)}{\sum_{j \in V} \exp(z_{t,j} / T)}$$
* **Temperature ($T$):** Divides raw logits $z_t$ before Softmax. Lower $T \to 0.1$ sharpens the probability peak on the single most likely token (deterministic/factual); higher $T > 0.9$ flattens the distribution (creative/diverse).
* **Top-$P$ (Nucleus Sampling):** Sorts tokens by descending probability and samples only from the minimal nucleus $V^{(p)}$ such that $\sum_{w \in V^{(p)}} P(w) \ge P$.

### 4.2 Context Engineering & Prompting Paradigms
* **System Prompt:** Establishes global persona, domain constraints, and output formatting schema.
* **Zero-Shot vs. Few-Shot (In-Context Learning):** Supplying $k$ input-output demonstrations shifts the Transformer's internal activation trajectory without updating model weights.
* **Chain-of-Thought (CoT):** Instructing the model to emit intermediate reasoning tokens before the final answer allocates extra sequential compute to multi-step logic.
        """,
        "task_prompt": "Experiment with the Temperature slider to observe how the Next-Token Softmax Probability Distribution changes in real time, then run Zero-Shot vs. Chain-of-Thought prompts in the LLM comparator.",
        "quiz": [
            {
                "q": "Mathematically, what happens to the Softmax next-token probability distribution as Temperature (T) approaches 0.1?",
                "options": [
                    "Probability mass concentrates heavily on the highest-logit token, making output nearly deterministic",
                    "Every token in the vocabulary becomes equally likely",
                    "The context window shrinks to 1 token",
                    "The model stops reading the system prompt"
                ],
                "ans": 0
            },
            {
                "q": "Supplying 3 labeled input-output pairs inside the prompt to guide formatting without retraining weights is called:",
                "options": [
                    "Few-Shot (In-Context) Learning",
                    "Gradient Backpropagation",
                    "Pre-training from scratch",
                    "Tokenization"
                ],
                "ans": 0
            },
            {
                "q": "Why does Chain-of-Thought ('Think step-by-step') significantly improve LLM accuracy on engineering and math problems?",
                "options": [
                    "It allows the autoregressive model to generate intermediate reasoning tokens that condition subsequent calculations",
                    "It increases GPU clock frequency",
                    "It disables the attention mechanism",
                    "It removes numbers from the prompt"
                ],
                "ans": 0
            }
        ]
    },
    "mod5_rag": {
        "id": "mod5_rag",
        "title": "5. Retrieval-Augmented Generation (RAG)",
        "subtitle": "5-Stage Pipeline: PDF Parsing, Sliding-Window Chunking, Embedding, ChromaDB Retrieval & Grounded Synthesis",
        "theory": r"""
### 5.1 Why RAG? Overcoming Parametric Hallucinations
Standard LLMs store knowledge solely in static weights $\theta$, leading to hallucinations on private institutional policies or post-cutoff data. **Retrieval-Augmented Generation (RAG)** couples a non-parametric vector index with a parametric generator:
$$P_{\text{RAG}}(y \mid x) \approx \sum_{d \in \text{Top-}K(x)} P_\eta(d \mid x)\, P_\theta(y \mid x, d)$$

### 5.2 The 5 Internal RAG Pipeline Stages
1. **Stage 1 — Document Ingestion & Sliding-Window Chunking:** Raw PDFs/documents are parsed and sliced into overlapping chunks of length $C$ with overlap $O$ (stride $S = C - O$) so sentences crossing boundaries preserve semantic continuity.
2. **Stage 2 — Dense Embedding Generation:** Each chunk $c_i$ is encoded via `all-MiniLM-L6-v2` into $v_i \in \mathbb{R}^{384}$.
3. **Stage 3 — Vector Database Indexing:** Tuples `(chunk_id, text, vector, metadata)` are stored in ChromaDB.
4. **Stage 4 — Query Encoding & Top-$K$ Retrieval:** User question $q \to v_q$ retrieves the Top-$K$ highest Cosine Similarity chunks.
5. **Stage 5 — Context Augmentation & Grounded Generation:** Retrieved chunks are injected into a strict system prompt instructing the LLM to cite chunk IDs and abstain if the answer is absent.
        """,
        "task_prompt": "Select one of the preloaded SAKEC PDFs (or upload your own PDF), tune Chunk Size & Overlap, inspect all 5 visual stages including the Chunk Similarity Bar Chart, and generate a grounded RAG response.",
        "quiz": [
            {
                "q": "What is the primary engineering reason for configuring a non-zero 'Chunk Overlap' (e.g., 40 characters) in Stage 1 of RAG?",
                "options": [
                    "To ensure sentences or concepts split across chunk boundaries retain context in adjacent chunks",
                    "To corrupt the PDF text",
                    "To bypass the embedding model",
                    "To reduce Cosine Similarity to zero"
                ],
                "ans": 0
            },
            {
                "q": "In Stage 4 of RAG, how does the system determine which document chunks are most relevant to the user's question?",
                "options": [
                    "It embeds the user question into the same 384-D vector space and computes Cosine Similarity against all chunk vectors",
                    "It picks the first 3 paragraphs alphabetically",
                    "It asks the user to type the chunk numbers manually",
                    "It deletes all chunks shorter than 500 words"
                ],
                "ans": 0
            },
            {
                "q": "How does Stage 5 (Augmented Prompt Construction) prevent the LLM from hallucinating?",
                "options": [
                    "By explicitly injecting the Top-K retrieved chunks into the prompt and instructing the LLM to answer strictly from that provided context",
                    "By setting Temperature to 2.5",
                    "By hiding the user question from the LLM",
                    "By turning off the vector database"
                ],
                "ans": 0
            }
        ]
    },
    "mod6_agentic": {
        "id": "mod6_agentic",
        "title": "6. Agentic AI & ReAct Tool Calling",
        "subtitle": "Autonomous Reasoning + Acting (ReAct) Loops, Deterministic Tool Schemas & Multi-Step Planning",
        "theory": r"""
### 6.1 From Passive Text Generators to Autonomous AI Agents
While a standard LLM maps `Prompt -> Text` in a single forward pass and a RAG system executes a fixed `Retrieve -> Generate` pipeline, an **Agentic AI System** uses the LLM as a dynamic **Reasoning Engine** inside a feedback loop (`ReAct`: Reasoning + Acting):
$$\text{User Goal} \longrightarrow \underbrace{\text{Thought}_t \longrightarrow \text{Action}_t(\text{Tool}, \text{JSON Args}) \longrightarrow \text{Observation}_t}_{t = 1, 2, \dots, T \text{ until goal satisfied}} \longrightarrow \text{Final Grounded Answer}$$

### 6.2 Core Components of an Agentic Architecture
1. **Planner / Reasoning Loop:** Decomposes complex compound requests into atomic sub-goals.
2. **Tool Registry & Structured Function Schemas:** Deterministic external tools (e.g., Python Math Evaluator, Vector Knowledge Base Search, EV Battery Range Simulator, Live Telemetry Sensor).
3. **Working Memory (Scratchpad):** Appends each `(Thought, Action, Observation)` tuple back into the context window so the agent can self-correct and synthesize verified results.
        """,
        "task_prompt": "Execute a compound multi-tool query in the Agentic ReAct Sandbox (combining arithmetic/EV battery calculation, campus policy lookup, and sensor telemetry) and inspect the step-by-step Thought -> Action -> Observation trace.",
        "quiz": [
            {
                "q": "In the ReAct (Reasoning + Acting) agent paradigm, what happens immediately after the agent emits an 'Action' (Tool Call)?",
                "options": [
                    "The external tool executes deterministically and returns an 'Observation' back into the agent's context scratchpad",
                    "The agent deletes its system prompt",
                    "The user's account is logged out",
                    "The neural network retrains its weights"
                ],
                "ans": 0
            },
            {
                "q": "Why should an Agentic AI system invoke a deterministic Calculator/Python tool for engineering math rather than relying on LLM token prediction?",
                "options": [
                    "Because autoregressive token prediction is probabilistic and prone to floating-point arithmetic errors, whereas a calculator tool is 100% exact",
                    "Because LLMs cannot output digits",
                    "Because tools increase hallucination rates",
                    "Because calculators do not require electricity"
                ],
                "ans": 0
            },
            {
                "q": "What distinguishes an Agentic RAG system from a Naive RAG pipeline?",
                "options": [
                    "An Agentic system can autonomously decide which tools to call, reformulate queries, combine calculator outputs with vector search, and iterate across multiple steps",
                    "Agentic RAG cannot read PDFs",
                    "Naive RAG uses more tools than Agentic RAG",
                    "Agentic RAG does not use an LLM"
                ],
                "ans": 0
            }
        ]
    }
}
