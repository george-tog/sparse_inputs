from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def ensure_output_dirs() -> None:
    """Create standard generated-output directories used by notebooks."""
    for path in [
        RESULTS_DIR,
        RESULTS_DIR / "GLUE",
        RESULTS_DIR / "linear_decode_Poisson",
        RESULTS_DIR / "linear_decode_poisson",
        RESULTS_DIR / "OB_PiC_comparison",
        FIGURES_DIR,
        FIGURES_DIR / "manuscript",
        FIGURES_DIR / "matched_conc",
    ]:
        path.mkdir(parents=True, exist_ok=True)
