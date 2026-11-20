from pathlib import Path

dirs = ["docs", "data/raw", "data/processed", "notebooks",
        "src/exoplanet", "configs", "results", "tests"]
for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)

Path("src/exoplanet/__init__.py").touch()
Path("requirements.txt").write_text(
    "numpy\npandas\nscikit-learn\nmatplotlib\ntensorflow\nlightkurve\njupyter\nipykernel\n"
)
with open(".gitignore", "a") as f:
    f.write("\ndata/raw/\ndata/processed/\n*.h5\n*.keras\n.venv/\n.ipynb_checkpoints/\n")