from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PPI_DIR = PROCESSED_DIR / "ppi"
GNN_DIR = PROCESSED_DIR / "gnn"

TP53_DIR = RAW_DIR / "tp53"
BIOGRID_DIR = RAW_DIR / "biogrid"
STRING_DIR = RAW_DIR / "string"

TP53_FILE = TP53_DIR / "tp53_mutations.csv"
BIOGRID_FILE = BIOGRID_DIR / "biogrid_all_4.4.243.tab3.txt"
STRING_LINKS_FILE = STRING_DIR / "string_links_human_v12.txt.gz"
STRING_INFO_CANDIDATES = (
    STRING_DIR / "string_protein_info_human_v12.txt",
    STRING_DIR / "9606.protein.info.v12.0.txt",
)
STRING_INFO_FILE = next(
    (path for path in STRING_INFO_CANDIDATES if path.exists()),
    STRING_INFO_CANDIDATES[0],
)

for directory in (RAW_DIR, PROCESSED_DIR, PPI_DIR, GNN_DIR, TP53_DIR, BIOGRID_DIR, STRING_DIR):
    directory.mkdir(parents=True, exist_ok=True)
