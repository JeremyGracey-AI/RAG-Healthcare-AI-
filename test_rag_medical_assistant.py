import importlib
import sys
import unittest
from unittest import mock


class ImportBehaviorTests(unittest.TestCase):
    def test_module_imports_without_optional_ml_dependencies(self):
        sys.modules.pop("rag_medical_assistant", None)

        original_import_module = importlib.import_module

        def guarded_import(name, package=None):
            optional_modules = (
                "chromadb",
                "fitz",
                "langchain",
                "llama_cpp",
                "sentence_transformers",
            )
            if name == optional_modules or name.startswith(optional_modules):
                raise ImportError(f"No module named {name!r}")
            return original_import_module(name, package)

        with mock.patch("importlib.import_module", side_effect=guarded_import):
            module = importlib.import_module("rag_medical_assistant")

        self.assertEqual(module.RAGConfig().retriever_k, 5)


class MainStartupTests(unittest.TestCase):
    def test_main_skips_heavy_setup_when_required_assets_are_missing(self):
        import rag_medical_assistant as rag

        with (
            mock.patch.object(rag.os.path, "exists", return_value=False),
            mock.patch.object(rag.logger, "warning") as warning,
            mock.patch.object(rag.logger, "info"),
        ):
            result = rag.main()

        self.assertEqual(result["status"], "missing_assets")
        self.assertEqual(
            result["missing_assets"],
            [
                "PDF data file: merck_manual.pdf",
                "Mistral model file: ./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            ],
        )
        warning.assert_any_call(
            "Full RAG pipeline cannot start because required assets are missing:"
        )
        warning.assert_any_call("  - %s", "PDF data file: merck_manual.pdf")
        warning.assert_any_call(
            "  - %s",
            "Mistral model file: ./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        )


class MistralLLMTests(unittest.TestCase):
    def test_missing_model_file_raises_file_not_found_before_dependency_import(self):
        import rag_medical_assistant as rag

        with mock.patch.object(rag.os.path, "exists", return_value=False):
            with self.assertRaises(FileNotFoundError) as context:
                rag.MistralLLM("/missing/model.gguf")

        self.assertEqual(str(context.exception), "Model not found at /missing/model.gguf")


if __name__ == "__main__":
    unittest.main()
