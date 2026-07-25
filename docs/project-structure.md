# CME Project Directory & Tech Stack Specification

This specification outlines the hybrid **Python/Rust** tech stack and provides a production-ready directory structure designed for the **Cognitive Memory Engine (CME)**. Pass this document directly to your coding agent to initialize the repository, generate boilerplates, or write a comprehensive `README.md`.

---

## 1. The Technology Stack

To build a high-performance, future-proof cognitive operating system, we utilize a **hybrid language architecture**:

```
[ LLM Client / API ] ➔ [ Python SDK / Orchestrator ] ➔ [ Rust Core Engine (FFI) ]
                             │                                  │
                    (Qiskit / Ocean SDK)              (Petgraph / Memory-Mapped)
                             ▼                                  ▼
                   [ Quantum Backends ]               [ Ultra-fast Graph Traversal ]
```

### Core Languages
*   **Python 3.12+ (The Orchestrator & Science Interface):**
    *   **Why we use it:** Python has the richest ecosystem for AI model integrations (OpenAI, Claude, Llama, DeepSeek) [13], mathematical modeling, and quantum computing.
    *   **Key Libraries:** 
        *   `fastapi` / `uvicorn` (For high-performance async API endpoints).
        *   `pydantic` (For strict data validation of Beliefs, Evidence, and Confidence metrics) [6].
        *   `qiskit` / `pennylane` (For gate-based Quantum Approximate Optimization Algorithm - QAOA) [11].
        *   `dwave-ocean-sdk` (For Quadratic Unconstrained Binary Optimization - QUBO and Quantum Annealing solvers) [10, 11].
*   **Rust (The Performance-Critical Core):**
    *   **Why we use it:** Memory management, pointer-chasing in large graph structures, and belief propagation are computationally heavy. Rust provides zero-cost abstractions, guaranteed thread-safety, and blazing-fast execution speeds [9].
    *   **How they link:** We use **PyO3** and **Maturin** to compile the core Rust database structures into a native Python module (`cme_core`). This gives developers a friendly Python API with native Rust execution speed.

### Database & Knowledge Storage Layer
*   **Graph Database (The Semantic Web):**
    *   **Choice:** **Memgraph** or **Neo4j** (via Bolt protocol) for persistent storage, paired with Rust's **petgraph** library for in-memory graph traversal during active reasoning cycles.
    *   **Why:** CME connects every belief into a dynamic Knowledge Graph [8, 9]. We need index-free adjacency to query relationships (e.g., *Quantum Computing ➔ Qubits*) without slowing down as the memory grows [9].
*   **Vector Database (The Evidence Library):**
    *   **Choice:** **Qdrant** or **pgvector (PostgreSQL)**.
    *   **Why:** Used by the Evidence Engine to store raw source snippets, documents, and historical contexts [6, 9]. When a belief is queried, the vector DB retrieves the matching evidence chunks to prove the claim [9].
*   **Relational Storage (Belief Registry & Logs):**
    *   **Choice:** **PostgreSQL** or **SQLite** (for lightweight local development).
    *   **Why:** Best for keeping track of structured belief records (Confidence Scores, Timestamps, Source metadata) [6].

---

## 2. Recommended Directory Structure

This structure separates the high-performance Rust backend from the modular Python engine, making it easy to scale, test, and containerize.

```text
cognitive-memory-engine/
├── .github/                     # CI/CD Workflows (Rust tests, Python linting)
├── cme-core/                    # --- RUST CORE ENGINE ---
│   ├── Cargo.toml               # Rust dependencies (pyo3, petgraph, rayon)
│   └── src/
│       ├── lib.rs               # PyO3 bindings exposing Rust to Python
│       ├── belief_graph.rs      # Fast graph node/edge management for Beliefs
│       ├── memory_index.rs      # Low-latency memory storage structures
│       └── reasoning_ops.rs     # Rust implementations of multi-hop propagation
│
├── cme_python/                  # --- PYTHON SDK & ORCHESTRATION ---
│   ├── __init__.py
│   ├── config.py                # Global settings (DB URIs, LLM API keys)
│   ├── main.py                  # FastAPI server entry point
│   │
│   ├── engines/                 # The Seven Cognitive Engines
│   │   ├── __init__.py
│   │   ├── belief.py            # Belief Extraction and schema modeling
│   │   ├── memory.py            # Persistent Organizational/Scientific Memory
│   │   ├── graph.py             # Python interface wrapper for the Rust Graph
│   │   ├── evidence.py          # Evidence Engine (Vector database connector)
│   │   ├── reasoning.py         # Conflicting belief detection & logic
│   │   ├── optimization.py      # Classical solvers (SciPy, CVXPY)
│   │   └── quantum_layer.py     # Quantum solvers (Qiskit, Ocean SDK, QAOA)
│   │
│   ├── clients/                 # LLM Connectors
│   │   ├── __init__.py
│   │   ├── openai_client.py
│   │   └── claude_client.py
│   │
│   └── utils/
│       └── text_processing.py
│
├── tests/                       # --- SYSTEM TESTING ---
│   ├── rust_tests/              # Rust unit tests
│   └── python_tests/            # Python integration and API tests
│
├── docker/                      # --- CONTAINERIZATION ---
│   ├── Dockerfile
│   └── docker-compose.yml       # Spins up Memgraph, Qdrant, PostgreSQL, and CME
│
├── pyproject.toml               # Poetry/Maturin configuration
├── requirements.txt             # Pip fallback dependencies
└── README.md                    # Project overview
```

---

## 3. Step-by-Step Development Process

To keep your team aligned and ensure steady progress, execute your build in these **four phases**:

### Phase 1: The Groundwork (Python Prototype)
*   **Goal:** Build a pure Python proof-of-concept of the first 4 engines (Belief, Memory, Graph, Evidence) [8, 9].
*   **Action:** Write the core `Pydantic` schemas for a Belief [6], configure a local `SQLite` database to track them [6], and connect a vector store to serve as your Evidence Engine [9].

### Phase 2: Optimization & Quantum Solvers
*   **Goal:** Implement the Optimization Engine and Quantum Layer [10].
*   **Action:** Write classical solvers using `SciPy` to run memory-retrieval and reasoning path selections [10]. Next, set up the **Optional Optimization Backend** using `dwave-ocean-sdk` or `Qiskit` so you can convert these selection paths into QUBO (Quadratic Unconstrained Binary Optimization) models [11].

### Phase 3: Rust Porting (High-Performance Acceleration)
*   **Goal:** Accelerate graph lookups and belief propagation [9].
*   **Action:** Build the `cme-core` Rust module. Move the heavy mathematical loops, graph traversals, and multi-hop reasoning algorithms from Python into Rust [9]. Compile using `Maturin` to bind it back into Python seamlessy.

### Phase 4: Integrations & API
*   **Goal:** Provide the "Plug-and-Play" infrastructure [13].
*   **Action:** Build the `FastAPI` server and client wrappers [13]. Create a middleware that intercepts queries from OpenAI, Claude, or Llama, processes them through CME, and appends grounded facts and proof before handing the query to the user [11, 13].
