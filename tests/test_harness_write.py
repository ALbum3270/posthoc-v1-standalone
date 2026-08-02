from __future__ import annotations

import asyncio

from open_deep_research.harness.write import (
    build_write_prompt,
    parse_report_citations,
    write_report,
)


class FakeWriteModel:
    def __init__(self, markdown: str) -> None:
        self.markdown = markdown
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> dict[str, object]:
        self.prompts.append(prompt)
        return {
            "content": self.markdown,
            "token_count": 17,
            "cost_usd": 0.25,
        }


def test_write_report_passes_all_material_once_without_report_template() -> None:
    assembled = "FIRST SOURCE BODY\n\nSECOND SOURCE BODY\n"
    model = FakeWriteModel("# Model-chosen heading\n\nA narrative claim.")

    report = asyncio.run(write_report(assembled, model_client=model))

    assert report.canonical_draft == (
        "# Model-chosen heading\n\nA narrative claim."
    )
    assert report.token_count == 17
    assert report.cost_usd == 0.25
    assert len(model.prompts) == 1
    assert assembled in model.prompts[0]
    assert "content,\nstructure, length, and headings" in model.prompts[0]
    assert "Introduction" not in build_write_prompt(assembled)
    assert "Conclusion" not in build_write_prompt(assembled)
    assert "citation protocol" not in model.prompts[0]
    assert "[^1]" not in model.prompts[0]
    assert '"quote"' not in model.prompts[0]
    assert "Do not add footnotes, citations" in model.prompts[0]
    assert "Do not reproduce note_id or source_id" in model.prompts[0]
    assert "Do not present any passage as a verbatim quotation" in (
        model.prompts[0]
    )
    assert "Do not add evidence, grounding, confidence" in model.prompts[0]


def test_write_prompt_keeps_report_language_bound_to_the_research_topic() -> None:
    assembled = "# Assembled research notes\n\nMostly English source material."
    topic = "FTX 客户资金去了哪里？"
    model = FakeWriteModel("# 报告\n\n正文。")

    asyncio.run(write_report(assembled, model_client=model, topic=topic))

    assert f"Research topic:\n{topic}" in model.prompts[0]
    assert "same primary natural language as the research topic" in (
        model.prompts[0]
    )
    assert "does not require covering" in model.prompts[0]


def test_parse_report_citations_returns_exact_triples_and_reports_issues() -> None:
    markdown = """\
# A model-chosen report

The first assertion is supported. [^1]

The second assertion has two sources. [^2][^3]

This assertion has no marker.

This assertion points nowhere. [^missing]

This assertion has a broken definition. [^broken]

## Sources

[^1]: {"quote":"Exact quote one.","url":"https://one.example/a"}
[^2]: {"quote":"Exact quote two.","url":"https://two.example/b"}
[^3]: {"quote":"Exact quote three.","url":"https://three.example/c"}
[^broken]: not-json
"""

    parsed = parse_report_citations(markdown)

    assert [
        (citation.claim, citation.quote, citation.url)
        for citation in parsed.citations
    ] == [
        (
            "The first assertion is supported.",
            "Exact quote one.",
            "https://one.example/a",
        ),
        (
            "The second assertion has two sources.",
            "Exact quote two.",
            "https://two.example/b",
        ),
        (
            "The second assertion has two sources.",
            "Exact quote three.",
            "https://three.example/c",
        ),
    ]
    assert [
        (issue.claim, issue.reason, issue.reference_ids)
        for issue in parsed.unresolved_claims
    ] == [
        ("This assertion has no marker.", "missing_reference", ()),
        (
            "This assertion points nowhere.",
            "unknown_reference",
            ("missing",),
        ),
        (
            "This assertion has a broken definition.",
            "malformed_reference_definition",
            ("broken",),
        ),
    ]


def test_parse_report_citations_never_throws_for_absent_citations() -> None:
    parsed = parse_report_citations("A completely uncited factual assertion.")

    assert parsed.citations == ()
    assert len(parsed.unresolved_claims) == 1
    assert parsed.unresolved_claims[0].reason == "missing_reference"
