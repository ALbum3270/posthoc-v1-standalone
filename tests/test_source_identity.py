from open_deep_research.graphrag.validation.sources import publisher_identity


def test_paths_and_www_variants_are_one_publisher() -> None:
    assert publisher_identity("https://www.example.com/a") == "example.com"
    assert publisher_identity("https://news.example.com/b") == "example.com"


def test_multi_label_public_suffix_keeps_the_publisher_label() -> None:
    assert publisher_identity("https://www.bbc.co.uk/news") == "bbc.co.uk"


def test_different_publishers_remain_independent() -> None:
    assert publisher_identity("https://reuters.com/a") != publisher_identity(
        "https://apnews.com/b"
    )


def test_document_id_is_a_conservative_fallback() -> None:
    assert publisher_identity(None, fallback="Doc-1") == "document:doc-1"
