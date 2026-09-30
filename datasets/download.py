import csv
import io
import urllib.request
from pathlib import Path

DATASETS = {
    "choices13k": {
        "url": "https://raw.githubusercontent.com/jcpeterson/choices13k/main/c13k_selections.csv",
        "filename": "c13k_selections.csv",
    },
    "cpc18": {
        "url": "https://raw.githubusercontent.com/jcpeterson/choices13k/main/cpc18.csv",
        "filename": "cpc18.csv",
    },
}


def download_dataset(name: str, output_dir: str = "datasets") -> Path:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(DATASETS)}")

    info = DATASETS[name]
    output_path = Path(output_dir) / info["filename"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"{name} already exists at {output_path}")
        return output_path

    print(f"Downloading {name} from {info['url']}...")
    with urllib.request.urlopen(info["url"]) as resp:
        data = resp.read().decode("utf-8")

    output_path.write_text(data)
    print(f"Saved to {output_path}")
    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download decision-making datasets")
    parser.add_argument("datasets", nargs="+", choices=list(DATASETS.keys()))
    parser.add_argument("--output-dir", default="datasets")
    args = parser.parse_args()

    for name in args.datasets:
        download_dataset(name, args.output_dir)


if __name__ == "__main__":
    main()
