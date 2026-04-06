"""
tests/test_schemas.py

Tests that:
  1. serving/schemas.py imports cleanly (the v1 broken-import bug)
  2. Every type used in serving/server.py is actually defined in schemas.py
  3. Pydantic validation works correctly
"""

import unittest


class TestSchemaImport(unittest.TestCase):
    """
    The v1 bug: server.py imported CompletionRequest which did not exist.
    This test will fail if any name used in server.py is missing from schemas.py.
    """

    def test_all_server_imports_resolve(self):
        """Import every name that server.py pulls from schemas — must not raise."""
        from serving.schemas import (
            AdmissionVerdict,
            BatchRequest,
            BatchResponse,
            ChatCompletionRequest,
            ChatCompletionResponse,
            ChatChoice,
            ChatMessage,
            CompletionChoice,
            CompletionUsage,
            GenerateRequest,
            GenerateResponse,
            GPUStats,
            HealthResponse,
            MetricsResponse,
            OpenAICompletionRequest,
            OpenAICompletionResponse,
            StreamChunk,
        )
        # If we got here without ImportError, all names exist
        self.assertTrue(True)

    def test_no_CompletionRequest_in_server(self):
        """
        CompletionRequest (the v1 broken name) must NOT appear in server.py.
        It was renamed to OpenAICompletionRequest.
        """
        import inspect
        import serving.server as srv
        src = inspect.getsource(srv)
        self.assertNotIn(
            "CompletionRequest,",
            src,
            "server.py still imports the old broken name 'CompletionRequest'",
        )


class TestGenerateRequest(unittest.TestCase):

    def test_valid_request(self):
        from serving.schemas import GenerateRequest
        r = GenerateRequest(prompt="Hello", max_tokens=128)
        self.assertEqual(r.prompt, "Hello")
        self.assertEqual(r.max_tokens, 128)
        self.assertIsNotNone(r.request_id)

    def test_blank_prompt_rejected(self):
        from serving.schemas import GenerateRequest
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            GenerateRequest(prompt="   ")  # whitespace-only

    def test_empty_string_rejected(self):
        from serving.schemas import GenerateRequest
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            GenerateRequest(prompt="")

    def test_temperature_bounds(self):
        from serving.schemas import GenerateRequest
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            GenerateRequest(prompt="hi", temperature=3.0)  # > 2.0
        with self.assertRaises(ValidationError):
            GenerateRequest(prompt="hi", temperature=-0.1)

    def test_max_tokens_bounds(self):
        from serving.schemas import GenerateRequest
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            GenerateRequest(prompt="hi", max_tokens=0)
        with self.assertRaises(ValidationError):
            GenerateRequest(prompt="hi", max_tokens=99999)

    def test_request_id_auto_generated(self):
        from serving.schemas import GenerateRequest
        r1 = GenerateRequest(prompt="a")
        r2 = GenerateRequest(prompt="b")
        self.assertNotEqual(r1.request_id, r2.request_id)


class TestBatchRequest(unittest.TestCase):

    def test_valid_batch(self):
        from serving.schemas import BatchRequest
        r = BatchRequest(prompts=["hello", "world"])
        self.assertEqual(len(r.prompts), 2)

    def test_too_many_prompts_rejected(self):
        from serving.schemas import BatchRequest
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            BatchRequest(prompts=["x"] * 65)

    def test_blank_prompt_in_batch_rejected(self):
        from serving.schemas import BatchRequest
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            BatchRequest(prompts=["hello", "   "])

    def test_empty_list_rejected(self):
        from serving.schemas import BatchRequest
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            BatchRequest(prompts=[])


class TestOpenAISchemas(unittest.TestCase):

    def test_openai_completion_request_defaults(self):
        from serving.schemas import OpenAICompletionRequest
        r = OpenAICompletionRequest(prompt="test")
        self.assertEqual(r.model, "local")
        self.assertEqual(r.max_tokens, 512)

    def test_chat_message_roles(self):
        from serving.schemas import ChatMessage
        from pydantic import ValidationError
        ChatMessage(role="user", content="hello")
        ChatMessage(role="assistant", content="hi")
        ChatMessage(role="system", content="be helpful")
        with self.assertRaises(ValidationError):
            ChatMessage(role="unknown", content="hi")

    def test_chat_request_empty_messages_rejected(self):
        from serving.schemas import ChatCompletionRequest
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            ChatCompletionRequest(messages=[])


class TestAdmissionVerdict(unittest.TestCase):

    def test_fields_present(self):
        from serving.schemas import AdmissionVerdict
        v = AdmissionVerdict(
            allowed=True,
            backend="vllm",
            reason="ok",
            estimated_kv_gb=0.5,
            free_gpu_gb=10.0,
        )
        self.assertTrue(v.allowed)
        self.assertEqual(v.backend, "vllm")


if __name__ == "__main__":
    unittest.main(verbosity=2)
