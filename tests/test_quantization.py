"""
tests/test_quantization.py

Tests for inference/quantization.py.

This file was listed in the v1 README but did not exist — that caused both
the doc drift and a missing test signal.  Every edge case from the review
is covered here, including the RuntimeError-before-ValueError bug.

All tests run without GPU, CUDA, transformers, or bitsandbytes installed.
"""

import sys
import types
import unittest


# ---------------------------------------------------------------------------
# Helpers to simulate missing packages without actually installing anything
# ---------------------------------------------------------------------------

def _remove_module(name: str):
    """Remove a module from sys.modules so imports fail naturally."""
    sys.modules.pop(name, None)
    # Also remove sub-modules
    to_remove = [k for k in sys.modules if k.startswith(name + ".")]
    for k in to_remove:
        del sys.modules[k]


def _install_stub(name: str, **attrs):
    """Install a minimal stub module so imports succeed."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildQuantizationConfig(unittest.TestCase):

    def setUp(self):
        # Make sure quantization module is re-imported fresh each test
        _remove_module("inference.quantization")
        _remove_module("inference")

    def _import(self):
        from inference.quantization import build_quantization_config
        return build_quantization_config

    # ── Valid methods that return None without any imports ──────────────────

    def test_none_method_returns_none(self):
        fn = self._import()
        self.assertIsNone(fn({"method": "none"}))

    def test_awq_returns_none(self):
        fn = self._import()
        self.assertIsNone(fn({"method": "awq"}))

    def test_gptq_returns_none(self):
        fn = self._import()
        self.assertIsNone(fn({"method": "gptq"}))

    def test_empty_dict_defaults_to_none(self):
        fn = self._import()
        # default method is "none"
        self.assertIsNone(fn({}))

    # ── Unknown method must raise ValueError BEFORE any import ─────────────

    def test_invalid_method_raises_ValueError(self):
        """
        This was the v1 bug: RuntimeError from ImportError fired before ValueError.
        Now ValueError is raised immediately after the method string check,
        before any import statement is executed.
        """
        fn = self._import()
        with self.assertRaises(ValueError) as ctx:
            fn({"method": "invalid"})
        self.assertIn("invalid", str(ctx.exception))
        self.assertIn("none", str(ctx.exception))  # shows valid options

    def test_unknown_method_is_ValueError_not_RuntimeError(self):
        fn = self._import()
        exc_type = None
        try:
            fn({"method": "gguf"})
        except Exception as e:
            exc_type = type(e)
        self.assertIs(exc_type, ValueError)

    def test_capitalised_invalid_method_raises_ValueError(self):
        fn = self._import()
        with self.assertRaises(ValueError):
            fn({"method": "INT4"})

    # ── 8bit / 4bit raise ImportError when transformers is absent ──────────

    def test_8bit_without_transformers_raises_ImportError(self):
        _remove_module("transformers")
        fn = self._import()
        with self.assertRaises(ImportError) as ctx:
            fn({"method": "8bit"})
        self.assertIn("bitsandbytes", str(ctx.exception).lower())

    def test_4bit_without_transformers_raises_ImportError(self):
        _remove_module("transformers")
        fn = self._import()
        with self.assertRaises(ImportError):
            fn({"method": "4bit"})

    # ── With transformers stub, 8bit and 4bit succeed ───────────────────────

    def test_8bit_with_stub_transformers(self):
        """Simulate transformers installed with a stub BitsAndBytesConfig."""
        try:
            import torch as real_torch
        except ImportError:
            self.skipTest("torch not installed — skipping stub test")

        class FakeBnBConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        _install_stub("transformers", BitsAndBytesConfig=FakeBnBConfig)
        sys.modules.setdefault("torch", real_torch)

        fn = self._import()
        result = fn({"method": "8bit"})
        self.assertIsNotNone(result)
        self.assertTrue(result.load_in_8bit)

    def test_4bit_with_stub_transformers(self):
        try:
            import torch as real_torch
        except ImportError:
            self.skipTest("torch not installed — skipping stub test")

        class FakeBnBConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        _install_stub("transformers", BitsAndBytesConfig=FakeBnBConfig)
        sys.modules.setdefault("torch", real_torch)

        fn = self._import()
        result = fn({"method": "4bit", "quant_type": "nf4", "double_quant": True})
        self.assertIsNotNone(result)
        self.assertTrue(result.load_in_4bit)
        self.assertEqual(result.bnb_4bit_quant_type, "nf4")
        self.assertTrue(result.bnb_4bit_use_double_quant)

    # ── Case insensitivity ──────────────────────────────────────────────────

    def test_method_case_insensitive(self):
        fn = self._import()
        self.assertIsNone(fn({"method": "AWQ"}))
        self.assertIsNone(fn({"method": "None"}))
        self.assertIsNone(fn({"method": "GPTQ"}))


class TestValidMethods(unittest.TestCase):
    """Verify the _VALID_METHODS constant is correct."""

    def test_valid_methods_set(self):
        _remove_module("inference.quantization")
        from inference.quantization import _VALID_METHODS
        self.assertIn("none", _VALID_METHODS)
        self.assertIn("4bit", _VALID_METHODS)
        self.assertIn("8bit", _VALID_METHODS)
        self.assertIn("awq", _VALID_METHODS)
        self.assertIn("gptq", _VALID_METHODS)
        # Should NOT contain invalid entries
        self.assertNotIn("gguf", _VALID_METHODS)
        self.assertNotIn("invalid", _VALID_METHODS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
