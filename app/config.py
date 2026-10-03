"""
Infrastructure, Model Specifications, and Hardware Detection Configuration.
"""

import os
import platform
import sys
import torch
import streamlit as st

from app.inference import load_inference_models


MODEL_OPTIONS = {
    "Qwen 3.5 (0.8B)": "Qwen/Qwen3.5-0.8B",
    "Llama 3.2 (1B Instruct)": "meta-llama/Llama-3.2-1B-Instruct",
    "Gemma 3 (1B IT)": "google/gemma-3-1b-it",
}


def get_model_specs(model_name: str):
    """
    Retrieve architecture and empirical holdout benchmark metrics for a model.
    """
    if "Llama" in model_name:
        return {
            "p_count": "1.23 B",
            "roc_val": "0.7450 (XGB) / 0.6929 (LR)",
            "pr_val": "0.7492 (XGB) / 0.6681 (LR)",
            "acc_val": "66.88% (XGB)",
            "ece_val": "0.0392",
            "arch": "LlamaForCausalLM (32 Layers, GQA)",
            "ctx_len": "128k Tokens (RoPE)",
        }
    elif "Gemma" in model_name:
        return {
            "p_count": "1.00 B",
            "roc_val": "0.7913 (XGB) / 0.7092 (LR)",
            "pr_val": "0.7914 (XGB) / 0.7054 (LR)",
            "acc_val": "71.50% (XGB)",
            "ece_val": "0.0410",
            "arch": "Gemma3ForCausalLM (GeGLU, RMSNorm)",
            "ctx_len": "32k Tokens",
        }
    else:
        return {
            "p_count": "0.80 B",
            "roc_val": "0.7688 (XGB) / 0.7079 (LR)",
            "pr_val": "0.7795 (XGB) / 0.7124 (LR)",
            "acc_val": "67.88% (XGB)",
            "ece_val": "0.0341",
            "arch": "Qwen2ForCausalLM (24 Layers, SwiGLU)",
            "ctx_len": "32k Tokens",
        }


@st.cache_resource
def get_hardware_info():
    """
    Dynamically detect host CPU and GPU hardware without hardcoded strings.
    """
    cpu_name = None
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except Exception:
            pass
    if not cpu_name:
        cpu_name = platform.processor() or platform.machine() or "Host Processor"

    cpu_cores = os.cpu_count() or 1
    cpu_display = f"{cpu_name} ({cpu_cores} vCPUs)"

    cuda_available = torch.cuda.is_available()
    gpu_name = None
    gpu_display = None
    gpu_count = 0
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "CUDA Device"
        try:
            total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            gpu_display = f"{gpu_name} ({total_mem_gb:.1f} GB)"
        except Exception:
            gpu_display = gpu_name

    return {
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "gpu_display": gpu_display,
        "gpu_count": gpu_count,
        "cpu_name": cpu_name,
        "cpu_display": cpu_display,
    }


@st.cache_resource(show_spinner="Initializing neural pipelines and cached weights...")
def get_cached_models(model_id: str = "Qwen/Qwen3.5-0.8B", target_device: str = "cuda:0"):
    """
    Load and persist models in Streamlit's resource cache across sessions and reruns.
    Supports Qwen3.5-0.8B, Llama 3.2 1B Instruct, and Gemma 3 1B IT.
    """
    effective_device = target_device if (torch.cuda.is_available() and "cuda" in target_device) else "cpu"
    return load_inference_models(
        device=effective_device,
        model_id=model_id,
    )
