import pytest

from mcp_multi_agent.faq import FaqIndex, load_entries, stem, tokenize


@pytest.fixture(scope="module")
def index():
    return FaqIndex(load_entries())


def test_all_faq_documents_are_loaded():
    entries = load_entries()

    assert len(entries) == 8
    assert all(entry.title and entry.text for entry in entries)


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("What are the opening hours?") == ["open", "hour"]


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("cancelled", "cancel"),
        ("cancellations", "cancel"),
        ("vaccines", "vaccin"),
        ("vaccine", "vaccin"),
        ("cat", "cat"),
    ],
)
def test_stem_normalizes_common_forms(word, expected):
    assert stem(word) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What time do you open on Saturday?", "opening-hours"),
        ("How often does a puppy need vaccine boosters?", "vaccination"),
        ("How long does a bath take?", "grooming"),
        ("my dog ate something toxic", "emergencies"),
        ("do I need to fast before a blood test", "exams"),
        ("can I pay in installments with a credit card", "payments"),
        ("I need to cancel my appointment", "cancellation"),
        ("what should I bring on the first visit", "first-visit"),
    ],
)
def test_the_right_document_ranks_first(index, question, expected):
    hits = index.search(question)

    assert hits[0].id == expected


def test_results_are_ordered_and_limited(index):
    hits = index.search("vaccine card appointment", top_k=2)

    assert len(hits) <= 2
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_unrelated_question_returns_nothing(index):
    assert index.search("quantum chromodynamics lagrangian") == []


def test_empty_query_returns_nothing(index):
    assert index.search("   ") == []
