"""
Session & Audit History Management for VERITAS.

Handles:
1. Session lifecycle: unique session ID generation, start timestamps, empty history initialization.
2. Detector activity recording: input text, result verdicts, confidence scores, timestamps, telemetry.
3. In-memory session containment: automatically discarded when the application session terminates.
4. Export and clear/reset utilities.
"""

import csv
import datetime
import io
import json
from typing import Any, Dict, List, Optional
import uuid

import streamlit as st


def generate_session_id() -> str:
    """Generate a clean, professional unique session identifier."""
    return f"SES-{uuid.uuid4().hex[:8].upper()}"


def init_session(force_new: bool = False) -> str:
    """
    Initialize a unique session if not already initialized or if forced.
    Ensures a unique session ID and an empty history list per session lifecycle.
    """
    if force_new or "session_id" not in st.session_state:
        st.session_state["session_id"] = generate_session_id()
        now = datetime.datetime.now()
        st.session_state["session_start_time"] = now.strftime("%I:%M:%S %p")
        st.session_state["session_start_date"] = now.strftime("%b %d, %Y")
        st.session_state["session_start_iso"] = now.isoformat()
        # Initial empty history list for the session
        st.session_state["history"] = []

    if "history" not in st.session_state or not isinstance(st.session_state["history"], list):
        st.session_state["history"] = []

    return st.session_state["session_id"]


def get_session_id() -> str:
    """Retrieve the current session identifier."""
    return st.session_state.get("session_id") or init_session()


def get_session_history() -> List[Dict[str, Any]]:
    """Retrieve the current session detection records."""
    if "history" not in st.session_state:
        init_session()
    return st.session_state.get("history", [])


def clear_session_history() -> None:
    """Discard all detection records from the current session."""
    st.session_state["history"] = []


def reset_session() -> str:
    """Terminate the current session and start a brand new session with empty history."""
    return init_session(force_new=True)


def calculate_confidence_score(risk_prob: float) -> float:
    """
    Calculate normalized classification confidence percentage (0.0 to 100.0%).
    Confidence represents model certainty in its verdict:
    - High confidence if probability is close to 0.0 (certainly faithful) or 1.0 (certainly hallucinated).
    - Lower confidence when probability is near decision boundary (0.5).
    """
    certainty = abs(float(risk_prob) - 0.5) * 2.0  # 0.0 at 0.5, 1.0 at 0.0 or 1.0
    # Map to 50% - 99.9% scale for intuitive display
    conf_pct = 50.0 + (certainty * 49.9)
    return round(conf_pct, 1)


def record_detector_activity(
    prompt: str,
    result: Dict[str, Any],
    context: Optional[str] = None,
    model_name: str = "Qwen 3.5 (0.8B)",
    threshold_low: float = 35.0,
    threshold_high: float = 65.0,
    is_evidence_mode: bool = False,
) -> Dict[str, Any]:
    """
    Record an inference detection activity into the current session's history.
    Appends input, result, confidence score, risk percentage, and timestamp.
    """
    session_id = get_session_id()
    now = datetime.datetime.now()

    # Extract multi-model risk probabilities from inference result
    xgb_val = float(
        result.get("predicted_risk_xgb", result.get("xgb_hallucination_probability", 0.0)) or 0.0
    )
    lr_val = float(
        result.get("predicted_risk_lr", result.get("lr_hallucination_probability", 0.0)) or 0.0
    )
    combined_val = float(
        result.get(
            "predicted_risk_combined",
            result.get("combined_hallucination_probability", (xgb_val + lr_val) / 2.0),
        ) or 0.0
    )

    xgb_risk_pct = round(xgb_val * 100.0, 1)
    lr_risk_pct = round(lr_val * 100.0, 1)
    combined_risk_pct = round(combined_val * 100.0, 1)

    # Combined posterior risk & calibrated confidence
    risk_pct = combined_risk_pct
    conf_pct = calculate_confidence_score(combined_val)

    # Classify tier based on operational thresholds
    if risk_pct < float(threshold_low):
        tier = "Faithful"
        tier_color = "emerald"
    elif risk_pct >= float(threshold_high):
        tier = "High Risk"
        tier_color = "rose"
    else:
        tier = "Indeterminate"
        tier_color = "amber"

    clean_prompt = str(prompt).strip()
    clean_context = str(context).strip() if context else ""
    generated_text = str(result.get("generated_response", "")).strip()

    internal_signals = result.get("internal_signals", {})
    num_tokens = int(internal_signals.get("num_tokens", 0) or 0)
    perplexity = float(internal_signals.get("perplexity", 0.0) or 0.0)
    total_latency_ms = float(result.get("total_latency_ms", 0.0) or 0.0)

    record_id = f"REC-{len(st.session_state.get('history', [])) + 1:03d}-{uuid.uuid4().hex[:4].upper()}"

    record: Dict[str, Any] = {
        "id": record_id,
        "session_id": session_id,
        "timestamp": now.strftime("%I:%M:%S %p"),
        "date": now.strftime("%Y-%m-%d"),
        "iso_timestamp": now.isoformat(),
        # Input data
        "input": {
            "prompt": clean_prompt,
            "context": clean_context,
            "is_evidence_mode": bool(is_evidence_mode),
        },
        # Legacy compatibility keys for existing dashboard references
        "prompt": clean_prompt[:50] + ("..." if len(clean_prompt) > 50 else ""),
        "full_prompt": clean_prompt,
        "full_context": clean_context,
        # Multi-model risk telemetry
        "xgb_risk_pct": xgb_risk_pct,
        "lr_risk_pct": lr_risk_pct,
        "combined_risk_pct": combined_risk_pct,
        "risk_pct": combined_risk_pct,
        "xgb_prob": xgb_val,
        "lr_prob": lr_val,
        "combined_prob": combined_val,
        "confidence_score": conf_pct,
        # Result data
        "result": {
            "tier": tier,
            "tier_color": tier_color,
            "risk_pct": combined_risk_pct,
            "xgb_risk_pct": xgb_risk_pct,
            "lr_risk_pct": lr_risk_pct,
            "combined_risk_pct": combined_risk_pct,
            "xgb_prob": xgb_val,
            "lr_prob": lr_val,
            "combined_prob": combined_val,
            "confidence_pct": conf_pct,
            "generated_response": generated_text,
            "risk_level": result.get("risk_level", tier.upper().replace(" ", "_")),
        },
        "model": model_name,
        "tier": tier,
        "time": now.strftime("%I:%M %p"),
        # Telemetry & performance
        "telemetry": {
            "latency_ms": round(total_latency_ms, 1),
            "tokens": num_tokens,
            "perplexity": round(perplexity, 2),
            "feature_vector_len": len(result.get("feature_vector", []) or []),
        },
    }

    if "history" not in st.session_state:
        st.session_state["history"] = []

    # Insert newest record at top
    st.session_state["history"].insert(0, record)

    # Cap at 100 entries per session to conserve memory
    if len(st.session_state["history"]) > 100:
        st.session_state["history"] = st.session_state["history"][:100]

    return record


def record_benchmark_activity(
    prompt: str,
    benchmark_results: Dict[str, Any],
    context: Optional[str] = None,
    threshold_low: float = 35.0,
    threshold_high: float = 65.0,
    is_evidence_mode: bool = False,
) -> Dict[str, Any]:
    """
    Record a multi-model benchmark evaluation suite into the current session's history as a single unified card.
    Consolidates answers, risk percentages, confidence scores, and telemetry across all evaluated models.
    """
    session_id = get_session_id()
    now = datetime.datetime.now()

    clean_prompt = str(prompt).strip()
    clean_context = str(context).strip() if context else ""

    models_list = list(benchmark_results.keys())
    model_summaries = []
    total_tokens = 0
    total_latency_ms = 0.0
    risk_values = []
    conf_values = []

    for m_name, res in benchmark_results.items():
        if "error" in res:
            continue
        xgb_val = float(res.get("predicted_risk_xgb", res.get("xgb_hallucination_probability", 0.0)) or 0.0)
        lr_val = float(res.get("predicted_risk_lr", res.get("lr_hallucination_probability", 0.0)) or 0.0)
        combined_val = float(res.get("predicted_risk_combined", res.get("combined_hallucination_probability", (xgb_val + lr_val) / 2.0)) or 0.0)

        xgb_risk_pct = round(xgb_val * 100.0, 1)
        lr_risk_pct = round(lr_val * 100.0, 1)
        comb_risk_pct = round(combined_val * 100.0, 1)
        conf_pct = calculate_confidence_score(combined_val)

        if comb_risk_pct < float(threshold_low):
            m_tier = "Faithful"
            m_tier_color = "emerald"
        elif comb_risk_pct >= float(threshold_high):
            m_tier = "High Risk"
            m_tier_color = "rose"
        else:
            m_tier = "Indeterminate"
            m_tier_color = "amber"

        int_sig = res.get("internal_signals", {})
        num_toks = int(int_sig.get("num_tokens", len(res.get("token_details", []))) or 0)
        ppl = float(int_sig.get("perplexity", 1.0) or 1.0)
        mean_ent = float(int_sig.get("mean_entropy", 0.0) or 0.0)
        lat_ms = float(res.get("total_latency_ms", 0.0) or 0.0)

        total_tokens += num_toks
        total_latency_ms += lat_ms
        risk_values.append(comb_risk_pct)
        conf_values.append(conf_pct)

        model_summaries.append({
            "model_name": m_name,
            "tier": m_tier,
            "tier_color": m_tier_color,
            "combined_risk_pct": comb_risk_pct,
            "xgb_risk_pct": xgb_risk_pct,
            "lr_risk_pct": lr_risk_pct,
            "confidence_pct": conf_pct,
            "num_tokens": num_toks,
            "latency_ms": round(lat_ms, 1),
            "perplexity": round(ppl, 2),
            "mean_entropy": round(mean_ent, 3),
            "generated_response": str(res.get("generated_response", "")).strip(),
            "res": res,
        })

    model_summaries.sort(key=lambda x: x["combined_risk_pct"])

    avg_risk = round(sum(risk_values) / len(risk_values), 1) if risk_values else 0.0
    avg_conf = round(sum(conf_values) / len(conf_values), 1) if conf_values else 75.0
    best_model_name = model_summaries[0]["model_name"] if model_summaries else "N/A"
    best_model_risk = model_summaries[0]["combined_risk_pct"] if model_summaries else 0.0

    if not model_summaries:
        overall_tier = "Failed"
        overall_tier_color = "rose"
    elif any(m["tier"] == "High Risk" for m in model_summaries):
        overall_tier = "High Risk"
        overall_tier_color = "rose"
    elif all(m["tier"] == "Faithful" for m in model_summaries):
        overall_tier = "Faithful"
        overall_tier_color = "emerald"
    else:
        overall_tier = "Indeterminate"
        overall_tier_color = "amber"

    record_id = f"BM-{len(st.session_state.get('history', [])) + 1:03d}-{uuid.uuid4().hex[:4].upper()}"

    record: Dict[str, Any] = {
        "id": record_id,
        "is_benchmark": True,
        "record_type": "benchmark",
        "session_id": session_id,
        "timestamp": now.strftime("%I:%M:%S %p"),
        "date": now.strftime("%Y-%m-%d"),
        "iso_timestamp": now.isoformat(),
        "input": {
            "prompt": clean_prompt,
            "context": clean_context,
            "is_evidence_mode": bool(is_evidence_mode),
        },
        "prompt": clean_prompt[:50] + ("..." if len(clean_prompt) > 50 else ""),
        "full_prompt": clean_prompt,
        "full_context": clean_context,
        "model": f"Benchmark ({len(models_list)} Models)",
        "models": models_list,
        "model_summaries": model_summaries,
        "best_model": best_model_name,
        "best_model_risk": best_model_risk,
        "risk_pct": avg_risk,
        "combined_risk_pct": avg_risk,
        "confidence_score": avg_conf,
        "tier": overall_tier,
        "tier_color": overall_tier_color,
        "time": now.strftime("%I:%M %p"),
        "telemetry": {
            "latency_ms": round(total_latency_ms, 1),
            "tokens": total_tokens,
            "models_count": len(models_list),
        },
        "result": {
            "tier": overall_tier,
            "tier_color": overall_tier_color,
            "risk_pct": avg_risk,
            "confidence_pct": avg_conf,
            "generated_response": f"Multi-model benchmark of {len(models_list)} engines. Best: {best_model_name} ({best_model_risk}% risk).",
        },
        "benchmark_results": benchmark_results,
    }

    if "history" not in st.session_state:
        st.session_state["history"] = []

    st.session_state["history"].insert(0, record)
    if len(st.session_state["history"]) > 100:
        st.session_state["history"] = st.session_state["history"][:100]

    return record


def delete_history_record(record_id: str) -> bool:
    """Delete a specific detection record from current session history."""
    history = st.session_state.get("history", [])
    initial_len = len(history)
    st.session_state["history"] = [r for r in history if r.get("id") != record_id]
    return len(st.session_state["history"]) < initial_len


def get_session_analytics() -> Dict[str, Any]:
    """Calculate aggregated telemetry for current session records."""
    history = get_session_history()
    total = len(history)

    if total == 0:
        return {
            "total_checks": 0,
            "faithful_count": 0,
            "high_risk_count": 0,
            "indeterminate_count": 0,
            "faithful_pct": 0.0,
            "high_risk_pct": 0.0,
            "avg_risk_pct": 0.0,
            "avg_confidence_pct": 0.0,
            "avg_latency_ms": 0.0,
        }

    faithful_cnt = sum(1 for r in history if r.get("tier") == "Faithful")
    high_risk_cnt = sum(1 for r in history if r.get("tier") == "High Risk")
    indet_cnt = total - faithful_cnt - high_risk_cnt

    avg_risk = sum(float(r.get("risk_pct", 0.0)) for r in history) / total
    avg_conf = sum(float(r.get("confidence_score", 75.0)) for r in history) / total

    latencies = [
        float(r.get("telemetry", {}).get("latency_ms", 0.0))
        for r in history
        if "telemetry" in r
    ]
    avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0

    return {
        "total_checks": total,
        "faithful_count": faithful_cnt,
        "high_risk_count": high_risk_cnt,
        "indeterminate_count": indet_cnt,
        "faithful_pct": round((faithful_cnt / total) * 100.0, 1),
        "high_risk_pct": round((high_risk_cnt / total) * 100.0, 1),
        "avg_risk_pct": round(avg_risk, 1),
        "avg_confidence_pct": round(avg_conf, 1),
        "avg_latency_ms": round(avg_latency, 1),
    }


def export_history_to_json() -> str:
    """Export current session history to formatted JSON string."""
    history = get_session_history()
    export_payload = {
        "session_id": get_session_id(),
        "exported_at": datetime.datetime.now().isoformat(),
        "total_records": len(history),
        "analytics": get_session_analytics(),
        "records": history,
    }
    return json.dumps(export_payload, indent=2, ensure_ascii=False)


def export_history_to_csv() -> str:
    """Export current session history to standard CSV string."""
    history = get_session_history()
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Record ID",
        "Session ID",
        "Timestamp",
        "Model",
        "Prompt",
        "Context",
        "Tier",
        "Combined Risk Pct",
        "XGB Risk Pct",
        "LR Risk Pct",
        "Confidence Score",
        "Tokens",
        "Latency ms",
        "Generated Response",
    ])

    for r in history:
        input_data = r.get("input", {})
        result_data = r.get("result", {})
        telemetry = r.get("telemetry", {})

        writer.writerow([
            r.get("id", ""),
            r.get("session_id", ""),
            r.get("timestamp", r.get("time", "")),
            r.get("model", ""),
            input_data.get("prompt", r.get("full_prompt", r.get("prompt", ""))),
            input_data.get("context", r.get("full_context", "")),
            r.get("tier", ""),
            r.get("combined_risk_pct", r.get("risk_pct", "")),
            r.get("xgb_risk_pct", result_data.get("xgb_risk_pct", "")),
            r.get("lr_risk_pct", result_data.get("lr_risk_pct", "")),
            r.get("confidence_score", ""),
            telemetry.get("tokens", ""),
            telemetry.get("latency_ms", ""),
            result_data.get("generated_response", ""),
        ])

    return output.getvalue()
