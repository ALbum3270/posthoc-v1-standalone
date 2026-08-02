from hashlib import sha256

import pytest

from open_deep_research.harness.source_spans import (
    SOURCE_SEGMENTATION_VERSION,
    SourceSegmentKind,
    build_source_span_registry,
    render_segmented_source,
    resolve_source_span,
)


def test_markdown_aware_registry_retains_exact_offsets_and_source_identity():
    source = (
        "# Heading\n\n"
        "First sentence. Second sentence.\n\n"
        "- List evidence\n"
        "- Other evidence\n\n"
        "| Name | Value |\n"
        "| --- | --- |\n"
        "| A | 1 |"
    )

    registry = build_source_span_registry(source)

    assert registry.source_text_sha256 == sha256(
        source.encode("utf-8")
    ).hexdigest()
    assert registry.segmentation_version == SOURCE_SEGMENTATION_VERSION
    assert [segment.segment_id for segment in registry.segments] == [
        f"S{index:06d}" for index in range(1, 8)
    ]
    assert [segment.kind for segment in registry.segments] == [
        SourceSegmentKind.HEADING,
        SourceSegmentKind.PARAGRAPH_SENTENCE,
        SourceSegmentKind.PARAGRAPH_SENTENCE,
        SourceSegmentKind.LIST_ITEM,
        SourceSegmentKind.LIST_ITEM,
        SourceSegmentKind.TABLE_ROW,
        SourceSegmentKind.TABLE_ROW,
    ]
    assert registry.segments[1].unit_id == registry.segments[2].unit_id
    assert registry.segments[3].unit_id != registry.segments[4].unit_id
    for segment in registry.segments:
        assert source[segment.start_char : segment.end_char] == segment.text

    rendered = render_segmented_source(source, registry)
    for segment in registry.segments:
        assert f"<{segment.segment_id}>{segment.text}" in rendered
        rendered = rendered.replace(f"<{segment.segment_id}>", "")
    assert rendered == source


def test_contiguous_sentence_range_in_one_paragraph_returns_exact_source_slice():
    source = "A happened. Therefore B followed. A third sentence."
    registry = build_source_span_registry(source)

    resolved = resolve_source_span(
        source,
        registry,
        start_segment_id="S000001",
        end_segment_id="S000002",
    )

    assert resolved.source_quote == "A happened. Therefore B followed."
    assert source[resolved.start_char : resolved.end_char] == resolved.source_quote


def test_v4_splits_prose_inside_one_list_item_but_preserves_its_unit():
    source = (
        "- First sentence in one list item. "
        "Second sentence in the same list item."
    )

    registry = build_source_span_registry(source)

    assert registry.segmentation_version.endswith("-v4")
    assert [segment.text for segment in registry.segments] == [
        "- First sentence in one list item.",
        "Second sentence in the same list item.",
    ]
    assert all(
        segment.kind is SourceSegmentKind.LIST_ITEM
        for segment in registry.segments
    )
    assert registry.segments[0].unit_id == registry.segments[1].unit_id
    resolved = resolve_source_span(
        source,
        registry,
        start_segment_id="S000001",
        end_segment_id="S000002",
    )
    assert resolved.source_quote == source
    assert resolved.segment_count == 2


def test_v4_splits_adjacent_cjk_sentences_without_whitespace():
    """Regression for finance-18's paragraph-wide claim annotations."""

    source = (
        "- **内部管理失控和风险失察**：SBF 的管理风格被员工形容为鲁莽且不负责任。"
        "公司缺乏有效的内部控制和风险管理体系。"
    )

    registry = build_source_span_registry(source)

    assert registry.segmentation_version.endswith("-v4")
    assert [segment.text for segment in registry.segments] == [
        "- **内部管理失控和风险失察**：SBF 的管理风格被员工形容为鲁莽且不负责任。",
        "公司缺乏有效的内部控制和风险管理体系。",
    ]


def test_v4_keeps_cjk_closing_quotes_and_titles_with_their_sentence():
    """CJK punctuation must not strand a closing quote in the next segment."""

    source = "她说：‘第一句。第二句。’随后引用《第三句。》"

    registry = build_source_span_registry(source)

    assert [segment.text for segment in registry.segments] == [
        "她说：‘第一句。",
        "第二句。’",
        "随后引用《第三句。》",
    ]
    assert all(
        not segment.text.startswith(("’", "」", "》"))
        for segment in registry.segments
    )


def test_v4_does_not_split_cjk_punctuation_inside_a_code_block():
    source = '```json\n{"message": "第一句。第二句。"}\n```'

    registry = build_source_span_registry(source)

    assert len(registry.segments) == 1
    assert registry.segments[0].kind is SourceSegmentKind.CODE_BLOCK
    assert registry.segments[0].text == source


def test_single_letter_name_initial_does_not_end_report_segment():
    source = (
        "He was replaced by John J. Ray III during the proceedings. "
        "A second sentence followed."
    )

    registry = build_source_span_registry(source)

    assert [segment.text for segment in registry.segments] == [
        "He was replaced by John J. Ray III during the proceedings.",
        "A second sentence followed.",
    ]
    assert all(
        not segment.text.endswith("John J.") for segment in registry.segments
    )


@pytest.mark.parametrize(
    ("start_id", "end_id", "message"),
    [
        ("S999999", "S000001", "unknown start_segment_id"),
        ("S000002", "S000001", "reversed"),
        ("S000001", "S000003", "Markdown unit boundary"),
    ],
)
def test_invalid_segment_ranges_are_rejected_without_clamping(
    start_id,
    end_id,
    message,
):
    source = "First sentence. Second sentence.\n\nDifferent paragraph."
    registry = build_source_span_registry(source)

    with pytest.raises(ValueError, match=message):
        resolve_source_span(
            source,
            registry,
            start_segment_id=start_id,
            end_segment_id=end_id,
        )


def test_registry_cannot_be_reused_after_source_text_changes():
    source = "Original source sentence."
    registry = build_source_span_registry(source)

    with pytest.raises(ValueError, match="does not match span registry hash"):
        resolve_source_span(
            "Changed source sentence.",
            registry,
            start_segment_id="S000001",
            end_segment_id="S000001",
        )
