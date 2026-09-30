"""BM25, dense and hybrid retrieval behind the common interface (src/retrieve.py)."""
import numpy as np
import pytest

from src.retrieve import (BM25Retriever, ChunkCorpus, DenseRetriever, Filters, HybridRetriever,
                          bm25_tokenize, reciprocal_rank_fusion, tiebreak_key)
from tests.fakes import FakeEmbedder, make_chunk

PROVENANCE = ("chunk_id", "season", "episode", "episode_title", "source_row_start", "source_row_end")


@pytest.fixture
def chunks():
    """Six chunks across three episodes; two have identical text to exercise ties."""
    return [
        make_chunk(1, 1, 0, 0, 20, "Carrie: I couldn't help but wonder about love in Manhattan.", ["Carrie"], "Sex and the City"),
        make_chunk(1, 1, 1, 18, 40, "Miranda: Brunch again? Samantha: Brunch is sacred.", ["Miranda", "Samantha"]),
        make_chunk(4, 13, 0, 0, 25, "Carrie: A plant! The man brought a living thing into my apartment.", ["Carrie"], "The Good Fight"),
        make_chunk(4, 13, 1, 20, 50, "Aidan: Closing at the end of next week. Carrie: The apartment is full of boxes.", ["Aidan", "Carrie"]),
        make_chunk(6, 3, 0, 0, 30, "Jack Berger: Come on in. Carrie: Nice.", ["Jack Berger", "Carrie"]),
        make_chunk(6, 3, 1, 25, 55, "Jack Berger: Come on in. Carrie: Nice.", ["Jack Berger", "Carrie"]),
    ]


@pytest.fixture
def corpus(chunks):
    return ChunkCorpus(chunks)


@pytest.fixture
def embedder():
    return FakeEmbedder(dim=64)


@pytest.fixture
def dense(corpus, chunks, embedder):
    vectors = embedder.embed_documents([c["text"] for c in chunks])
    return DenseRetriever(corpus, vectors, corpus.ids, embedder)


def ids(results) -> list[str]:
    return [r.chunk_id for r in results]


def test_bm25_tokenizer_splits_on_apostrophes_and_punctuation():
    assert bm25_tokenize("Aidan's apartment -- it's FULL!") == ["aidan", "s", "apartment", "it", "s", "full"]


def test_bm25_idf_never_rises_with_document_frequency(corpus):
    index = BM25Retriever(corpus).index
    doc_freq = {term: sum(term in doc for doc in index.doc_freqs) for term in index.idf}
    for a in index.idf:
        for b in index.idf:
            if doc_freq[a] < doc_freq[b]:
                assert index.idf[a] >= index.idf[b], (a, b)
    assert index.idf["carrie"] == 0.0              # in 5 of 6 chunks: carries no weight


def test_bm25_leaves_out_chunks_matching_only_zero_idf_words(corpus):
    assert BM25Retriever(corpus).search("Carrie", k=10) == []       # 'carrie' is in 5 of 6 chunks: IDF 0


def test_bm25_ranks_the_chunk_with_the_rare_term_first_and_skips_non_matches(corpus):
    results = BM25Retriever(corpus).search("living plant", k=10)
    assert ids(results) == ["s04e13-test-000"]                      # no other chunk shares a query term
    assert results[0].rank == 1 and results[0].score > 0


def test_bm25_prefers_more_matching_terms(corpus):
    results = BM25Retriever(corpus).search("apartment boxes closing", k=2)
    assert ids(results) == ["s04e13-test-001", "s04e13-test-000"]


def test_results_obey_k_and_number_ranks_from_one(corpus, dense):
    for retriever in (BM25Retriever(corpus), dense, HybridRetriever(corpus, [BM25Retriever(corpus), dense])):
        for k in (0, 1, 3, 100):
            results = retriever.search("Carrie apartment", k)
            assert len(results) <= k
            assert [r.rank for r in results] == list(range(1, len(results) + 1))
    assert len(dense.search("anything at all", 100)) == len(corpus.ids)   # dense always has k candidates


def by_tiebreak(*chunk_ids: str) -> list[str]:
    """The expected order of exactly tied chunks."""
    return sorted(chunk_ids, key=tiebreak_key)


def test_exact_ties_follow_the_fixed_tiebreak_order(corpus, dense):
    twins = ("s06e03-test-000", "s06e03-test-001")                  # identical text, identical scores
    assert ids(BM25Retriever(corpus).search("come on in nice", k=2)) == by_tiebreak(*twins)
    dense_results = dense.search("Jack Berger: Come on in. Carrie: Nice.", k=2)
    assert dense_results[0].score == dense_results[1].score
    assert ids(dense_results) == by_tiebreak(*twins)


def test_tie_order_does_not_depend_on_position_in_the_file(chunks, embedder):
    for order in (chunks, list(reversed(chunks))):
        corpus = ChunkCorpus(order)
        vectors = embedder.embed_documents([c["text"] for c in order])
        dense = DenseRetriever(corpus, vectors, corpus.ids, embedder)
        assert ids(BM25Retriever(corpus).search("come on in nice", k=2)) == by_tiebreak("s06e03-test-000", "s06e03-test-001")
        assert ids(dense.search("Jack Berger: Come on in. Carrie: Nice.", k=2)) == by_tiebreak("s06e03-test-000", "s06e03-test-001")


def test_repeated_searches_give_identical_rankings(corpus, dense):
    hybrid = HybridRetriever(corpus, [BM25Retriever(corpus), dense])
    for retriever in (BM25Retriever(corpus), dense, hybrid):
        assert retriever.search("brunch with Samantha", 5) == retriever.search("brunch with Samantha", 5)
    rebuilt = HybridRetriever(corpus, [BM25Retriever(corpus), dense])
    assert rebuilt.search("brunch with Samantha", 5) == hybrid.search("brunch with Samantha", 5)


def test_dense_scores_are_cosine_similarities(corpus, chunks, dense, embedder):
    results = dense.search(chunks[2]["text"], k=1)
    assert results[0].chunk_id == chunks[2]["chunk_id"]
    assert results[0].score == pytest.approx(1.0, abs=1e-6)          # a text is most similar to itself
    assert all(-1.0 - 1e-9 <= r.score <= 1.0 + 1e-9 for r in dense.search("love", k=6))


def test_dense_refuses_an_index_that_does_not_match_the_chunks(corpus, chunks, embedder):
    vectors = embedder.embed_documents([c["text"] for c in chunks])
    with pytest.raises(ValueError):
        DenseRetriever(corpus, vectors, list(reversed(corpus.ids)), embedder)
    with pytest.raises(ValueError):
        DenseRetriever(corpus, vectors[:3], corpus.ids, embedder)


def test_provenance_survives_retrieval_unchanged(corpus, chunks, dense):
    by_id = {c["chunk_id"]: c for c in chunks}
    hybrid = HybridRetriever(corpus, [BM25Retriever(corpus), dense])
    for retriever in (BM25Retriever(corpus), dense, hybrid):
        for result in retriever.search("Carrie apartment brunch", 6):
            chunk = by_id[result.chunk_id]
            assert all(getattr(result, key) == chunk[key] for key in PROVENANCE)
            assert list(result.speakers) == chunk["speakers"] and result.text == chunk["text"]
            assert result.method == retriever.name


def test_filters_restrict_every_retriever(corpus, dense):
    hybrid = HybridRetriever(corpus, [BM25Retriever(corpus), dense])
    only_s4e13 = Filters(season=4, episode=13)
    with_aidan = Filters(speakers=("Aidan", "Carrie"))
    for retriever in (BM25Retriever(corpus), dense, hybrid):
        assert {(r.season, r.episode) for r in retriever.search("Carrie", 10, only_s4e13)} <= {(4, 13)}
        assert ids(retriever.search("apartment boxes", 10, with_aidan)) == ["s04e13-test-001"]


def test_rrf_adds_reciprocal_ranks_and_records_components():
    fused = reciprocal_rank_fusion({"bm25": ["a", "b", "c"], "dense": ["c", "a"]}, rrf_k=60)
    assert fused["a"][0] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"][0] == pytest.approx(1 / 62)
    assert fused["c"] == (pytest.approx(1 / 63 + 1 / 61), {"bm25": 3, "dense": 1})


class StubRetriever:
    """Returns a fixed ranking, for testing fusion in isolation."""

    def __init__(self, name, corpus, ranking):
        self.name, self.corpus, self.ranking = name, corpus, ranking

    def search(self, query, k, filters=None):
        return [self.corpus.result(self.corpus.position[c], 0.0, r, self.name)
                for r, c in enumerate(self.ranking[:k], 1)]


def test_hybrid_rewards_agreement_and_breaks_exact_ties_by_the_fixed_order(corpus):
    a, b, c, d = corpus.ids[:4]
    hybrid = HybridRetriever(corpus, [StubRetriever("bm25", corpus, [d, a, c]),
                                      StubRetriever("dense", corpus, [a, c, b])], rrf_k=60, candidates=3)
    results = hybrid.search("q", k=4)
    assert ids(results) == [a, c, d, b]          # a: 1/62+1/61 > c: 1/63+1/62 > d: 1/61 > b: 1/63
    assert results[0].components == {"bm25": 2, "dense": 1}
    assert results[2].components == {"bm25": 1}
    tie = HybridRetriever(corpus, [StubRetriever("bm25", corpus, [d]), StubRetriever("dense", corpus, [b])],
                          rrf_k=60, candidates=1)
    assert ids(tie.search("q", k=2)) == by_tiebreak(b, d)   # both score 1/61


def test_duplicate_chunk_ids_are_rejected(chunks):
    with pytest.raises(ValueError):
        ChunkCorpus(chunks + [chunks[0]])
