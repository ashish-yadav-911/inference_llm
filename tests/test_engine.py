"""
tests/test_engine.py

Engine unit tests — no GPU, no real model.
torch is available in this environment so we can test the real memory path.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock


def _minimal_config(backend="vllm", quant="none", device="cpu") -> dict:
    return {
        "model": {"path": "test-model", "dtype": "float16", "trust_remote_code": False},
        "quantization": {"method": quant},
        "hardware": {
            "device": device,
            "num_gpus": 1,
            "gpu_memory_utilization": 0.90,
            "max_memory": None,
            "offload_folder": "/tmp/offload",
        },
        "generation": {
            "max_new_tokens": 32,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "max_model_len": 2048,
        },
        "server": {"port": 8000},
        "logging": {"level": "DEBUG"},
        "backend": backend,
    }


class TestEngineProperties(unittest.TestCase):

    def test_initial_state(self):
        from inference.engine import InferenceEngine
        e = InferenceEngine(_minimal_config())
        self.assertFalse(e.is_ready)
        self.assertEqual(e.model_id, "test-model")
        self.assertEqual(e.quantization, "none")

    def test_assert_loaded_raises_before_load(self):
        from inference.engine import InferenceEngine
        e = InferenceEngine(_minimal_config())
        with self.assertRaises(RuntimeError, msg="should raise before load"):
            e._assert_loaded()

    def test_assert_loaded_passes_after_mock_load(self):
        from inference.engine import InferenceEngine
        e = InferenceEngine(_minimal_config())
        e._loaded = True
        e._backend_name = "vllm"
        e._assert_loaded()  # should not raise


class TestBnBConfigBuilding(unittest.TestCase):
    """Test the static _build_bnb_config method without real transformers."""

    def setUp(self):
        # Remove transformers so fallback path is tested
        sys.modules.pop("transformers", None)

    def test_none_method_returns_none(self):
        from inference.engine import InferenceEngine
        result = InferenceEngine._build_bnb_config({"method": "none"})
        self.assertIsNone(result)

    def test_awq_returns_none(self):
        from inference.engine import InferenceEngine
        result = InferenceEngine._build_bnb_config({"method": "awq"})
        self.assertIsNone(result)

    def test_gptq_returns_none(self):
        from inference.engine import InferenceEngine
        result = InferenceEngine._build_bnb_config({"method": "gptq"})
        self.assertIsNone(result)

    def test_4bit_without_transformers_returns_none_with_warning(self):
        # When transformers is missing, _build_bnb_config warns and returns None
        from inference.engine import InferenceEngine
        result = InferenceEngine._build_bnb_config({"method": "4bit"})
        self.assertIsNone(result)  # degrades gracefully


class TestMemoryStats(unittest.TestCase):

    def test_memory_stats_cpu_fallback(self):
        from inference.engine import InferenceEngine
        import torch
        e = InferenceEngine(_minimal_config())
        # On a machine without CUDA, should return cpu entry
        if not torch.cuda.is_available():
            stats = e.memory_stats()
            self.assertEqual(len(stats), 1)
            self.assertEqual(stats[0]["name"], "cpu")
            self.assertEqual(stats[0]["allocated_gb"], 0)

    def test_memory_stats_returns_list(self):
        from inference.engine import InferenceEngine
        e = InferenceEngine(_minimal_config())
        stats = e.memory_stats()
        self.assertIsInstance(stats, list)
        self.assertGreater(len(stats), 0)
        self.assertIn("free_gb", stats[0])
        self.assertIn("total_gb", stats[0])


class TestChatTemplate(unittest.TestCase):

    def _engine_with_tokenizer(self, has_template: bool):
        from inference.engine import InferenceEngine

        e = InferenceEngine(_minimal_config())
        e._loaded = True

        mock_tok = MagicMock()
        if has_template:
            mock_tok.apply_chat_template.return_value = "<|user|>hi<|assistant|>"
        else:
            mock_tok.apply_chat_template.side_effect = Exception("no template")
        e._tokenizer = mock_tok
        return e

    def test_chat_template_applied_when_available(self):
        e = self._engine_with_tokenizer(has_template=True)
        from serving.schemas import ChatMessage
        msgs = [ChatMessage(role="user", content="hi")]
        result = e.apply_chat_template(msgs)
        self.assertIn("hi", result)

    def test_chat_template_fallback_when_missing(self):
        e = self._engine_with_tokenizer(has_template=False)
        from serving.schemas import ChatMessage
        msgs = [
            ChatMessage(role="system", content="be helpful"),
            ChatMessage(role="user", content="hello"),
        ]
        result = e.apply_chat_template(msgs)
        self.assertIn("hello", result)
        self.assertIn("Human:", result)
        self.assertIn("System:", result)

    def test_assistant_prefix_appended(self):
        e = self._engine_with_tokenizer(has_template=False)
        from serving.schemas import ChatMessage
        msgs = [ChatMessage(role="user", content="test")]
        result = e.apply_chat_template(msgs)
        self.assertTrue(result.strip().endswith("Assistant:"))


class TestShutdown(unittest.TestCase):

    def test_shutdown_clears_state(self):
        from inference.engine import InferenceEngine
        e = InferenceEngine(_minimal_config())
        e._loaded = True
        e._model = MagicMock()
        e._tokenizer = MagicMock()
        e.shutdown()
        self.assertIsNone(e._model)
        self.assertIsNone(e._tokenizer)
        self.assertFalse(e._loaded)


if __name__ == "__main__":
    unittest.main(verbosity=2)
