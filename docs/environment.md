# Environment & Hardware Documentation

## Overview
This document records the exact software runtime, package versions, and hardware configuration for the project **"Catching an LLM Lying: A Lightweight Classifier for Real-Time Hallucination Detection from Internal Generation Signals"**.

---

## Hardware and System Configuration

| Component | Specification |
| :--- | :--- |
| **Operating System** | Microsoft Windows 11 Home Single Language (64-bit, Version 10.0.26200) |
| **CPU Architecture** | x86_64 |
| **Primary GPU** | Intel(R) Iris(R) Xe Graphics (Driver Version: 32.0.101.5542) |
| **Discrete NVIDIA GPU** | None detected |
| **CUDA Available** | No (System operates in optimized CPU mode) |
| **CUDA Version** | N/A |

---

## Python & Core ML Dependency Versions

| Package | Installed Version | Purpose |
| :--- | :--- | :--- |
| **Python** | `3.13.14` | Core runtime |
| **PyTorch (`torch`)** | `2.14.0+cpu` | Tensor operations and neural inference |
| **Transformers** | `5.17.0` | Tokenizer and LLM logit/entropy extraction |
| **Sentence-Transformers**| `6.0.1` | Dense embeddings for semantic similarity |
| **Datasets** | `5.0.1` | Benchmark dataset loading and management |
| **scikit-learn** | `1.9.1` | Logistic Regression, calibration, metrics |
| **XGBoost** | `3.4.1` | Gradient-boosted decision tree baseline |
| **FAISS (`faiss-cpu`)** | `1.15.1` | Fast dense vector similarity search |
| **NumPy** | `2.5.3` | Numerical computing |
| **Pandas** | `3.0.5` | Tabular data analysis |
| **SciPy** | `1.18.1` | Statistical calculations & entropy metrics |
| **Streamlit** | `1.64.0` | Interactive demonstration UI |
| **Rank-BM25** | `0.2.2` | Sparse lexical evidence retrieval |
| **Evaluate** | `0.4.6` | Evaluation metrics suite |
| **PyTest** | `9.1.1` | Unit test execution |

---

## Virtual Environment Management

The project uses an isolated virtual environment located in `.venv/` at the root of the project.

### Virtual Environment Location
```text
C:\Users\HP\Desktop\Catching-an-LLM-Lying\.venv
```

### Activation Instructions

#### 1. Windows PowerShell
From the project root:
```powershell
.\.venv\Scripts\Activate.ps1
```

> **Note on Execution Policy:**
> If PowerShell blocks script execution, run this once in an elevated PowerShell or user scope:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

#### 2. Windows Command Prompt (cmd.exe)
From the project root:
```cmd
.venv\Scripts\activate.bat
```

#### 3. Deactivation (both shells)
```cmd
deactivate
```

---

## Verification Commands
To verify the environment and hardware acceleration at any time, run:

```bash
# Verify all library imports and core capabilities
python tests/test_environment.py

# Verify GPU/CPU tensor execution
python tests/test_gpu.py

# Run full test suite
pytest tests/
```
