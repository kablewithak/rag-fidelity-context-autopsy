from __future__ import annotations

import pytest

from rag_lab.tokenizers import TokenizationError, UnicodeCodePointTokenCounter


def test_offline_diagnostic_counter_reports_its_contract_and_round_trips_multilingual_text() -> None:
    counter = UnicodeCodePointTokenCounter()
    text = "Réinitialisez le mot de passe. Sicherheitsvorfall bestätigt."

    token_ids = counter.encode(text)

    assert counter.name == "diagnostic:unicode_codepoint_v1"
    assert counter.count(text) == len(token_ids)
    assert counter.decode(token_ids) == text
    assert counter.count(text) > 0


def test_offline_diagnostic_counter_rejects_invalid_code_point() -> None:
    counter = UnicodeCodePointTokenCounter()

    with pytest.raises(TokenizationError, match="valid Unicode code point"):
        counter.decode([0x110000])
