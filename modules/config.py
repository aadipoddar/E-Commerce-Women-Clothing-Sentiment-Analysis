"""Central configuration for the Women's E-Commerce Clothing Reviews study.

Every path, seed and column name used anywhere in the project lives here so the
notebook and the helper modules can never drift apart.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
FIGURES = ROOT / "figures"
PAPER_FIGURES = ROOT / "paper_figures"
RESULTS = ROOT / "results"
MODULES = ROOT / "modules"

for _d in (DATASET_DIR, FIGURES, PAPER_FIGURES, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

CSV_NAME = "Womens Clothing E-Commerce Reviews.csv"
KAGGLE_SLUG = "nicapotato/womens-ecommerce-clothing-reviews"

# ---------------------------------------------------------------- columns
COL_TITLE = "Title"
COL_TEXT = "Review Text"
COL_RATING = "Rating"
COL_RECOMMEND = "Recommended IND"
COL_AGE = "Age"
COL_FEEDBACK = "Positive Feedback Count"
CAT_COLS = ["Division Name", "Department Name", "Class Name"]

TEXT_COL = "text"        # engineered: Title + " " + Review Text
TARGET = "sentiment"     # engineered: 1 = positive (4-5), 0 = negative (1-2)

NEUTRAL_RATING = 3       # dropped - see WORKFLOW.md

# ---------------------------------------------------------------- splits
SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15          # taken from the remaining 85%

# ---------------------------------------------------------------- features
TFIDF_WORD = dict(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50_000)
TFIDF_CHAR = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                  sublinear_tf=True, max_features=50_000)
SVD_COMPONENTS = 300

# ---------------------------------------------------------------- tuning
N_TRIALS = 30
TUNING_METRIC = "macro_f1"

# ---------------------------------------------------------------- plotting
DPI = 300
FIGSIZE = (10, 6)
PALETTE = {"negative": "#c0392b", "positive": "#27ae60"}
CLASS_NAMES = ["negative", "positive"]


def excel(name: str) -> Path:
    """Path helper for a results workbook."""
    return RESULTS / name


def figure(name: str, paper: bool = False) -> Path:
    """Path helper for a figure file."""
    return (PAPER_FIGURES if paper else FIGURES) / name
