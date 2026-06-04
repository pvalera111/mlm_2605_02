# mlm_2605_02
# RHEL Expert Intelligence Engine (mlm_2605_02)

An autonomous, locally deployed generative-search AI pipeline (RAG architecture) optimized for **Red Hat Enterprise Linux 9.6 (Plow)**. The system translates complex, conversational natural language infrastructure queries into exact, production-ready bash commands using an offline, fine-tuned **DistilBert** neural network core. Operating entirely within a secure, air-gapped production environment, the engine eliminates external API dependencies and vendor lock-in. By restructuring standard 768-dimensional token embeddings through localized domain adaptation, the system eliminates out-of-vocabulary (OOV) ambiguities and accurately maps unpredictable human descriptions (e.g., system bottlenecks, container faults, firewall blocks) directly onto hardcoded expert administration frameworks with sub-millisecond execution latency.

---

## 🔬 Core System Philosophy & Operations

This pipeline bridges the gap between unstructured human engineering requests and absolute terminal actions without making external internet API calls. The architecture operates entirely on-premise, making it secure for air-gapped production datacenters. It relies on a two-phase engine model:

### 1. The Fine-Tuning Stage (`run_vector.py`)
The pipeline runs an aggressive **Domain Adaptation** phase using **Masked Language Modeling (MLM)**. It reads official technical documentation (`data/data.txt`) gathered dynamically from system `man` pages (such as `grubby`, `tcpdump`, `podman`, and `semanage`). 
* **Data Sanitization**: The injection module automatically strips terminal interface backspace anomalies (`\x08` control layout artifacts) to feed only clean enterprise strings into the network.
* **Semantic Anchor Mapping**: It correlates raw engineering contexts with a hardened **Expert Decision Matrix** (`data/base_knowledge.py`), randomly masking 15% of the operational language tokens.
* **Hardware Acceleration**: The architecture compiles its neural computational graphs directly onto multi-core Intel Xeon platforms, utilizing 90 engineering threads. It automatically leverages low-level instruction steps (`AVX512`, `AMX_TILE`, `AMX_INT8`) alongside XLA (Accelerated Linear Algebra) JIT compilation to optimize memory processing.
* **Convergence Optimization**: A custom `TargetLossStopping` callback handles cross-entropy minimization, immediately pausing cluster operations and freezing the weights matrix the millisecond the semantic loss falls below `0.06`.

### 2. The Runtime Retrieval Engine (`ask_expert.py`)
Once trained, the analytical engine loads the custom weights into standard memory (RAM) for sub-millisecond querying execution.
* **Noise Filtration**: When a user inputs an infrastructure problem, the engine immediately strips out conversational stop-words (`what`, `how`, `show`, `can i`, etc.) to isolate pure technical intent.
* **Mean Pooling Vectorization**: The clean query text passes through the DistilBert core model layers, projecting the intent into an exact 768-dimensional mathematical embedding vector space.
* **Deterministic Evaluation Block**: The runtime environment cross-references an external json schema (`data/routing_rules.json`) and calculates the **Cosine Similarity** by running a high-speed matrix dot product between the query vector and pre-indexed database tensors. The closest intent in the `rhel_expert_matrix` is retrieved instantly and displayed alongside its real-time confidence rating.

---

## 🧭 Visual System Pipeline Architecture

```text
                                [ USER INPUT QUERY ]
                                         │
                                         ▼
                           ┌───────────────────────────┐
                           │ get_mean_pooling_vector() │
                           │  (Stop-Words Filtration)  │
                           └─────────────┬─────────────┘
                                         │
                                         ▼
                           ┌───────────────────────────┐
                           │ Matrix Dot Product (RAM)  │
                           │ Cosine Similarity Search  │
                           └─────────────┬─────────────┘
                                         │
                                         ▼
                        🎯 SEMANTIC MATCH ──► 💻 TARGET COMMAND
```

---

## 🤖 Core Engine Specifications

```text
======================================================================
  🤖 CORE ARCHITECTURE     : DistilBert [Dual-Interface Neural Network]
  📊 TOTAL WEIGHTS         : ~66,000,000 FP32 Neurons
  📐 EMBEDDING DIMENSION   : 768 Vector Space
  🚀 ACCELERATION          : Intel Xeon Platform [90 Dedicated Threads]
  ⚡ RETRIEVAL PATTERN     : Matrix Dot Product [RAM Semantic Vector Search]
  📂 ACTIVE CONTEXT TAGS   : SYSTEMD | SECURITY | STORAGE | PODMAN | SELINUX
======================================================================
  📖 KNOWLEDGE DOMAINS INTEGRATED INTO THE LAYER:
  ├── [🌐 Network & Security] ── IP Tables, ARP Cache, TCP Sockets, Tcpdump
  ├── [🛡️ Firewalls & SELinux] ── Packet Filter Rules, Contexts, Policy Enforcement
  ├── [👁️ Debugging & Tracing] ── Kernel Probes, Kprobes, eBPF & Bpftrace
  ├── [📦 Container Isolation] ── Podman Microservices, Volumes, Pods Layout
  ├── [⏳ Sessions & Processes] ── Terminal Multiplexers, Tmux, Screen, Nohup
  ├── [💾 File System Layouts] ── Block Devices, Sizing Hierarchy, Lsblk
  └── [📊 Kernel Subsystems] ── /proc Core Dumps, Meminfo, Cpuinfo, Uptime
======================================================================
```

---

## 🧠 Deep Architectural Insights

### Weights Initialization & PyTorch-to-TensorFlow Interoperability
During the system boot sequence, the Hugging Face engine prints a routine confirmation notice indicating a highly efficient cross-framework translation under the hood. The baseline pre-trained DistilBert weights (approx. 66 million parameters distributed across a 768-dimensional embedding space) are natively stored in the optimized PyTorch format. 

Upon executing the fine-tuning module, the pipeline dynamically maps and streams these weights directly into TensorFlow layer primitives in local RAM. This guarantees 100% data integrity, allowing the system to leverage pre-existing linguistic representations without requiring manual cross-compilation of the core neural nodes.

### Corpus Injection & Domain Adaptation
The system leverages a localized **Domain Adaptation** approach through the injection of custom technical language lines extracted from raw RHEL documentation feeds (`man` pages corpus). 

When the training module initializes, it purges low-level terminal interface backspace anomalies (such as `\x08` layout artifacts) and structures raw operating system behaviors into explicit textual tokens. This process allows the offline neural net core to align abstract natural language expressions (e.g., *"sniffing packets"*, *"logout survival"*, *"sandboxed app"*) with rigid operational Linux key structures. 

As a result, the deep semantic layers adjust to recognize specific IT infrastructure patterns, drastically reducing the structural cross-entropy loss and stabilizing the mathematical dot-product projections.

---

## 📁 Repository Directory Structure

```text
.
├── base_model/              # Offline base pre-trained DistilBert weights & configs
│   ├── config.json          # Model network architecture configuration
│   ├── (DO NOT LOG TO GIT)  # Large tensor checkpoint matrix layers (*.safetensors)
│   └── tokenizer.json       # Vocabulary token translation matrix
├── data/                    # Dataset storage for fine-tuning & matrices
│   ├── base_knowledge.py    # Structured Expert Decision Matrix (Intent -> Solution)
│   ├── routing_rules.json   # Deterministic hard-routing keyword constraints configuration
│   └── data.txt             # Raw Linux Man Pages & external technical corpus
├── output/                  # Final fine-tuned production-ready weights
│   └── expert_rhel_doc_vector/
├── run_vector.py            # Fine-tuning module (Masked Language Modeling, 120 epochs)
└── ask_expert.py            # High-speed runtime RAM search query engine
```

---

## 🛠 *Critical* Dependency Setup & Weights Acquisition Guide

This repository contains only the mathematical logic, custom dataset models, and runtime pipelines. **Large weight tensors (~260MB binary distributions) must be seeded manually** to establish absolute network isolation from external public networks.

### Phase 1: Environment Provisioning & Environment Locks
Ensure Python 3.9+ and the standard framework primitives required to build the dual-interface RAG graph are active on the host machine:
```bash
sudo dnf groupinstall "Development Tools" -y
sudo dnf install python3-devel python3-pip -y
pip3 install tensorflow transformers numpy keras
```

### Phase 2: Manual Acquisition of the Vanilla Baseline Weights
To deploy the engine completely offline (Air-Gapped environments), you must populate the `base_model/` directory with vanilla DistilBert configuration vectors.

1. **Download Baseline Weights**:
   From a workstation with internet connectivity, download the baseline assets from Hugging Face (`distilbert-base-uncased` repository):
   * `config.json`
   * `tokenizer.json`
   * `model.safetensors` (or `tf_model.h5` depending on frame compatibility layouts)

2. **Populate Local Path Directory**:
   Create the directory layout inside your project root folder and drop the assets into place:
   ```bash
   mkdir -p base_model data output
   # [Move downloaded files into base_model/ here]
   ```

# RHEL Expert Intelligence Engine (mlm_2605_02)

An autonomous, locally deployed generative-search AI pipeline (RAG architecture) optimized for **Red Hat Enterprise Linux 9.6 (Plow)**. The system translates complex, conversational natural language infrastructure queries into exact, production-ready bash commands using an offline, fine-tuned **DistilBert** neural network core. Operating entirely within a secure, air-gapped production environment, the engine eliminates external API dependencies and vendor lock-in. By restructuring standard 768-dimensional token embeddings through localized domain adaptation, the system eliminates out-of-vocabulary (OOV) ambiguities and accurately maps unpredictable human descriptions (e.g., system bottlenecks, container faults, firewall blocks) directly onto hardcoded expert administration frameworks with sub-millisecond execution latency.

---

## 🔬 Core System Philosophy & Operations

This pipeline bridges the gap between unstructured human engineering requests and absolute terminal actions without making external internet API calls. The architecture operates entirely on-premise, making it secure for air-gapped production datacenters. It relies on a two-phase engine model:

### 1. The Fine-Tuning Stage (`run_vector.py`)
The pipeline runs an aggressive **Domain Adaptation** phase using **Masked Language Modeling (MLM)**. It reads official technical documentation (`data/data.txt`) gathered dynamically from system `man` pages (such as `grubby`, `tcpdump`, `podman`, and `semanage`). 
* **Data Sanitization**: The injection module automatically strips terminal interface backspace anomalies (`\x08` control layout artifacts) to feed only clean enterprise strings into the network.
* **Semantic Anchor Mapping**: It correlates raw engineering contexts with a hardened **Expert Decision Matrix** (`data/base_knowledge.py`), randomly masking 15% of the operational language tokens.
* **Hardware Acceleration**: The architecture compiles its neural computational graphs directly onto multi-core Intel Xeon platforms, utilizing 90 engineering threads. It automatically leverages low-level instruction steps (`AVX512`, `AMX_TILE`, `AMX_INT8`) alongside XLA (Accelerated Linear Algebra) JIT compilation to optimize memory processing.
* **Convergence Optimization**: A custom `TargetLossStopping` callback handles cross-entropy minimization, immediately pausing cluster operations and freezing the weights matrix the millisecond the semantic loss falls below `0.06`.

### 2. The Runtime Retrieval Engine (`ask_expert.py`)
Once trained, the analytical engine loads the custom weights into standard memory (RAM) for sub-millisecond querying execution.
* **Noise Filtration**: When a user inputs an infrastructure problem, the engine immediately strips out conversational stop-words (`what`, `how`, `show`, `can i`, etc.) to isolate pure technical intent.
* **Mean Pooling Vectorization**: The clean query text passes through the DistilBert core model layers, projecting the intent into an exact 768-dimensional mathematical embedding vector space.
* **Deterministic Evaluation Block**: The runtime environment cross-references an external json schema (`data/routing_rules.json`) and calculates the **Cosine Similarity** by running a high-speed matrix dot product between the query vector and pre-indexed database tensors. The closest intent in the `rhel_expert_matrix` is retrieved instantly and displayed alongside its real-time confidence rating.

---

## 🧭 Visual System Pipeline Architecture

```text
                                [ USER INPUT QUERY ]
                                         │
                                         ▼
                           ┌───────────────────────────┐
                           │ get_mean_pooling_vector() │
                           │  (Stop-Words Filtration)  │
                           └─────────────┬─────────────┘
                                         │
                                         ▼
                           ┌───────────────────────────┐
                           │ Matrix Dot Product (RAM)  │
                           │ Cosine Similarity Search  │
                           └─────────────┬─────────────┘
                                         │
                                         ▼
                        🎯 SEMANTIC MATCH ──► 💻 TARGET COMMAND
```

---

## 🤖 Core Engine Specifications

```text
======================================================================
  🤖 CORE ARCHITECTURE     : DistilBert [Dual-Interface Neural Network]
  📊 TOTAL WEIGHTS         : ~66,000,000 FP32 Neurons
  📐 EMBEDDING DIMENSION   : 768 Vector Space
  🚀 ACCELERATION          : Intel Xeon Platform [90 Dedicated Threads]
  ⚡ RETRIEVAL PATTERN     : Matrix Dot Product [RAM Semantic Vector Search]
  📂 ACTIVE CONTEXT TAGS   : SYSTEMD | SECURITY | STORAGE | PODMAN | SELINUX
======================================================================
  📖 KNOWLEDGE DOMAINS INTEGRATED INTO THE LAYER:
  ├── [🌐 Network & Security] ── IP Tables, ARP Cache, TCP Sockets, Tcpdump
  ├── [🛡️ Firewalls & SELinux] ── Packet Filter Rules, Contexts, Policy Enforcement
  ├── [👁️ Debugging & Tracing] ── Kernel Probes, Kprobes, eBPF & Bpftrace
  ├── [📦 Container Isolation] ── Podman Microservices, Volumes, Pods Layout
  ├── [⏳ Sessions & Processes] ── Terminal Multiplexers, Tmux, Screen, Nohup
  ├── [💾 File System Layouts] ── Block Devices, Sizing Hierarchy, Lsblk
  └── [📊 Kernel Subsystems] ── /proc Core Dumps, Meminfo, Cpuinfo, Uptime
======================================================================
```

---

## 🧠 Deep Architectural Insights

### Weights Initialization & PyTorch-to-TensorFlow Interoperability
During the system boot sequence, the Hugging Face engine prints a routine confirmation notice indicating a highly efficient cross-framework translation under the hood. The baseline pre-trained DistilBert weights (approx. 66 million parameters distributed across a 768-dimensional embedding space) are natively stored in the optimized PyTorch format. 

Upon executing the fine-tuning module, the pipeline dynamically maps and streams these weights directly into TensorFlow layer primitives in local RAM. This guarantees 100% data integrity, allowing the system to leverage pre-existing linguistic representations without requiring manual cross-compilation of the core neural nodes.

### Corpus Injection & Domain Adaptation
The system leverages a localized **Domain Adaptation** approach through the injection of custom technical language lines extracted from raw RHEL documentation feeds (`man` pages corpus). 

When the training module initializes, it purges low-level terminal interface backspace anomalies (such as `\x08` layout artifacts) and structures raw operating system behaviors into explicit textual tokens. This process allows the offline neural net core to align abstract natural language expressions (e.g., *"sniffing packets"*, *"logout survival"*, *"sandboxed app"*) with rigid operational Linux key structures. 

As a result, the deep semantic layers adjust to recognize specific IT infrastructure patterns, drastically reducing the structural cross-entropy loss and stabilizing the mathematical dot-product projections.

---

## 📁 Repository Directory Structure

```text
.
├── base_model/              # Offline base pre-trained DistilBert weights & configs
│   ├── config.json          # Model network architecture configuration
│   ├── (DO NOT LOG TO GIT)  # Large tensor checkpoint matrix layers (*.safetensors)
│   └── tokenizer.json       # Vocabulary token translation matrix
├── data/                    # Dataset storage for fine-tuning & matrices
│   ├── base_knowledge.py    # Structured Expert Decision Matrix (Intent -> Solution)
│   ├── routing_rules.json   # Deterministic hard-routing keyword constraints configuration
│   └── data.txt             # Raw Linux Man Pages & external technical corpus
├── output/                  # Final fine-tuned production-ready weights
│   └── expert_rhel_doc_vector/
├── run_vector.py            # Fine-tuning module (Masked Language Modeling, 120 epochs)
└── ask_expert.py            # High-speed runtime RAM search query engine
```

---

## 🛠 *Critical* Dependency Setup & Weights Acquisition Guide

This repository contains only the mathematical logic, custom dataset models, and runtime pipelines. **Large weight tensors (~260MB binary distributions) must be seeded manually** to establish absolute network isolation from external public networks.

### Phase 1: Environment Provisioning & Environment Locks
Ensure Python 3.9+ and the standard framework primitives required to build the dual-interface RAG graph are active on the host machine:
```bash
sudo dnf groupinstall "Development Tools" -y
sudo dnf install python3-devel python3-pip -y
pip3 install tensorflow transformers numpy keras
```

### Phase 2: Manual Acquisition of the Vanilla Baseline Weights
To deploy the engine completely offline (Air-Gapped environments), you must populate the `base_model/` directory with vanilla DistilBert configuration vectors.

1. **Download Baseline Weights**:
   From a workstation with internet connectivity, download the baseline assets from Hugging Face (`distilbert-base-uncased` repository):
   * `config.json`
   * `tokenizer.json`
   * `model.safetensors` (or `tf_model.h5` depending on frame compatibility layouts)

2. **Populate Local Path Directory**:
   Create the directory layout inside your project root folder and drop the assets into place:
   ```bash
   mkdir -p base_model data output
   # [Move downloaded files into base_model/ here]
   ```


