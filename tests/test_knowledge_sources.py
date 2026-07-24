"""Tests for optional engineering knowledge sources in chat."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from app.knowledge_sources import build_knowledge_context
from app.schemas import CADChatRequest


class KnowledgeSourceTests(unittest.TestCase):
    def test_build_knowledge_context_deduplicates_selected_sources(self) -> None:
        context, display_names = build_knowledge_context(
            ["kiss_agent", "fair_explorer", "kiss_agent"]
        )

        self.assertEqual(display_names, ["KISS Agent", "FAIR Explorer"])
        self.assertEqual(context.count("KISS Agent:"), 1)
        self.assertIn("FAIR Explorer:", context)
        self.assertIn("Do not claim that a live record", context)

    def test_chat_request_rejects_unknown_knowledge_source(self) -> None:
        with self.assertRaises(ValidationError):
            CADChatRequest(
                message="Size this bushing",
                prompt="Create a rubber bushing",
                knowledge_sources=["unknown_agent"],
            )


if __name__ == "__main__":
    unittest.main()
