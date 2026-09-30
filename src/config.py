"""Every tunable setting for the pipeline lives in this one file."""
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "SATC_all_lines.csv"          # read-only source, not in the repo
KAGGLE_DATASET = "snapcrack/every-sex-and-the-city-script/versions/3"   # fetched by download_data.py
RAW_CSV_SHA256 = "3572c0e7851600080bafb40ae691696983caa4946f5079e3093b19d23d734047"   # version 3's file
EPISODES_CSV = ROOT / "data" / "meta" / "episodes.csv"           # season, episode, episode_title
SPEAKER_ALIASES_JSON = ROOT / "data" / "meta" / "speaker_aliases.json"
PROCESSED_DIR = ROOT / "data" / "processed"
LINES_JSONL = PROCESSED_DIR / "lines.jsonl"
PARSE_REPORT_JSON = PROCESSED_DIR / "parse_report.json"
SCENES_JSONL = PROCESSED_DIR / "scenes_inferred.jsonl"
CHUNKS_DIR = PROCESSED_DIR / "chunks"                            # one .jsonl + .manifest.json per chunk set
INDEX_DIR = ROOT / "index"                                       # cached dense embeddings
EVAL_FILE = ROOT / "eval" / "eval.json"
EVAL_RESULTS_DIR = ROOT / "eval" / "results"

# --- Parsing ---------------------------------------------------------------
# A row like "- Hello? - Carrie, it's Stanford." holds two speaker turns and is split.
# The dataset's label can belong to either turn (here it belongs to the second), so by
# default no turn inherits it and every part gets speaker=None. Set True to give the
# label to the first turn instead.
SPLIT_TURNS_KEEP_LABEL_ON_FIRST = False

# --- Embedding model: one model for every chunk size in the primary benchmark ---
# Chunk size is the experimental variable; the embedder must stay fixed and must read every
# chunk in full. BGE-M3 accepts 8192 tokens, so 256-, 512- and 1024-token chunks all fit.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"   # pinned Hugging Face commit
EMBED_MAX_TOKENS = 8192      # configured context incl. special tokens; longer input raises, never truncates
EMBED_BATCH_SIZE = 4
EMBED_DEVICE = "cpu"

# --- Token-based chunking (the primary retrieval unit) ---------------------
TOKENIZER = EMBEDDING_MODEL                  # chunk sizes are counted in the embedder's own tokens
TOKENIZER_REVISION = EMBEDDING_REVISION      # ("regex" = dependency-free approximation, tests only)
CHUNK_SIZE = 512                             # default for `python -m src.chunk` (whole lines only)
CHUNK_OVERLAP = 64                           # max tokens of trailing lines repeated in the next chunk
CHUNK_CONFIGS = ((256, 32), (512, 64), (1024, 128))   # the primary experiment's (size, overlap) pairs
UNKNOWN_SPEAKER = "(unknown)"                # how speaker=None lines are shown in chunk text

# --- Retrieval -------------------------------------------------------------
BM25_K1 = 1.5
BM25_B = 0.75
# rank_bm25 sets the IDF of terms in more than half the chunks to EPSILON x (average IDF). Its
# default 0.25 gives "the"/"you"/"Carrie" an IDF of ~1.3-1.6 on this corpus, more than terms in
# just under half the chunks (IDF near 0). 0.0 clamps them to zero, so IDF never rises with
# document frequency (the textbook max(0, idf) form).
BM25_EPSILON = 0.0
RRF_K = 60                   # Reciprocal Rank Fusion constant: score = sum 1 / (RRF_K + rank)
HYBRID_CANDIDATES = 30       # how deep each retriever's ranking goes into the fusion
DEFAULT_K = 8                # results shown by `src.cli search`

# --- Evaluation --------------------------------------------------------------
EVAL_K_VALUES = (1, 5, 10)   # Hit@k and Recall@k cut-offs; MRR is computed over the top max(k)
EVAL_COMPARE_K = 5           # cut-off used to label per-query wins and losses between retrievers


# --- Heuristic scene grouping (optional metadata, never ground truth) -------
SCENE_DETECTION = True          # False: chunks carry no inferred_scenes field; nothing else changes
SCENE_WINDOW = 6                # lines compared on each side of a gap
SCENE_MAX_SIMILARITY = 0.25     # rarity-weighted speaker-set Jaccard at or below this marks a turnover
SCENE_MIN_LINES = 6             # no inferred scene shorter than this many lines
# Narration-style openers that usually start a new scene (matched at the start of a line).
SCENE_TRANSITION_PATTERN = (
    r"^(?:(?:later|earlier) that (?:day|night|evening|afternoon|morning|week)\b"
    r"|later on\b|meanwhile\b"
    r"|that (?:same )?(?:day|night|evening|afternoon|morning|weekend)\b"
    r"|the (?:next|following) (?:day|night|morning|evening|afternoon|week|weekend)\b"
    r"|next (?:morning|day|night|evening|week)\b"
    r"|(?:a|one|two|three|four|five|a few|several) (?:hours?|days?|weeks?|months?) later\b"
    r"|across town\b"
    r"|(?:on )?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday) "
    r"(?:morning|afternoon|evening|night)\b)"
)
