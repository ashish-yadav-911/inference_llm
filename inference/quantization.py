from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Valid method names — checked before any import is attempted
_VALID_METHODS = frozenset({"none", "8bit", "4bit", "awq", "gptq"}) # non appendable


def build_quantization_config(quant_cfg: dict):

    method = quant_cfg.get("method", "none").lower().strip()

    # --- Validate first, import second ---
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown quantization method: '{method}'. "
            f"Valid options: {sorted(_VALID_METHODS)}"
        )

    if method == "none":
        logger.info("Quantization: none (full precision FP16/BF16)")
        return None

    if method in ("awq", "gptq"):
        logger.info(
            "Quantization: %s — model is pre-quantized; BitsAndBytesConfig not required", method
        )
        return None

    # method is "8bit" or "4bit" — now import BnB
    try:
        from transformers import BitsAndBytesConfig  # type: ignore
        import torch
    except ImportError as exc:
        raise ImportError(
            "transformers and bitsandbytes are required for 8-bit/4-bit quantization. "
            "Install with: pip install transformers bitsandbytes"
        ) from exc

    if method == "8bit":
        logger.info("Quantization: 8-bit INT8 via bitsandbytes (LLM.int8)")
        return BitsAndBytesConfig(load_in_8bit=True)

    # method == "4bit"
    compute_dtype_str = quant_cfg.get("compute_dtype", "float16")
    compute_dtype = getattr(torch, compute_dtype_str, torch.float16)
    quant_type = quant_cfg.get("quant_type", "nf4")
    double_quant = quant_cfg.get("double_quant", True)

    logger.info(
        "Quantization: 4-bit NF4 | compute_dtype=%s | double_quant=%s",
        compute_dtype_str, double_quant,
    )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=double_quant,
    )


# ---------------------------------------------------------------------------
# Offline quantization jobs
# ---------------------------------------------------------------------------

def quantize_awq(model_path: str, output_path: str, calib_data: str = "pileval") -> None:
    """
    model_path : str
        HuggingFace model id or local path.
    output_path : str
        Directory where the AWQ model will be saved.
    calib_data : str
        Dataset name for calibration (default: pileval).
    """
    logger.info("AWQ quantization: %s → %s (calib=%s)", model_path, output_path, calib_data)
    try:
        from awq import AutoAWQForCausalLM  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise ImportError("autoawq required: pip install autoawq") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoAWQForCausalLM.from_pretrained(model_path)

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info("AWQ model saved → %s", output_path)


def quantize_gptq(
    model_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 128,
    calib_data: str = "wikitext2",
) -> None:
    """
    model_path : str
        HuggingFace model id or local path.
    output_path : str
        Directory to save the GPTQ model.
    bits : int
        Quantization bits (3 or 4).
    group_size : int
        GPTQ group size. 128 is standard.
    calib_data : str
        Calibration dataset.
    """
    logger.info("GPTQ quantization: %s → %s (bits=%d, group=%d)", model_path, output_path, bits, group_size)
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError("auto-gptq and datasets required: pip install auto-gptq datasets") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    quant_config = BaseQuantizeConfig(bits=bits, group_size=group_size, desc_act=False)
    model = AutoGPTQForCausalLM.from_pretrained(model_path, quant_config)

    data = load_dataset(calib_data, "wikitext-2-raw-v1", split="train")
    examples = [tokenizer(ex["text"]) for ex in data.select(range(128))]
    model.quantize(examples)
    model.save_quantized(output_path, use_safetensors=True)
    tokenizer.save_pretrained(output_path)
    logger.info("GPTQ model saved → %s", output_path)


def quantize_bitsandbytes(
    model_path: str,
    output_path: str,
    method: str = "4bit",
) -> None:

    logger.info("bitsandbytes quantization: %s → %s (method=%s)", model_path, output_path, method)
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    bnb_config = build_quantization_config({"method": method})
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info("bitsandbytes model saved → %s", output_path)
