import click
from pathlib import Path
from tqdm import tqdm
from ..io import load_hf_dataset_config_or_none, load_runs_failsave
import gzip


@click.argument("corpus-directory", type=Path)
def export_corpus(corpus_directory: Path) -> int:
    """Export a corpus into a pre-defined directory so that everything is self-contained."""
    import ir_datasets
    from tira.ir_datasets_loader import IrDatasetsLoader
    # todo create a shared method

    files_to_remove = ["documents.jsonl", "queries.jsonl", "queries.xml", "metadata.json"]
    for f in files_to_remove:
        if (corpus_directory / f).is_file():
            (corpus_directory / f).unlink()

    runs = load_runs_failsave(corpus_directory / "runs")
    if len(runs) < 2:
        raise ValueError(f"Not enough runs, this is likely an error. I found {len(runs)} runs.")

    all_docs = set()

    for run in runs:
        for reference in run.references:
            all_docs.add(reference)
        for i in run.responses:
            for reference in i.citations:
                all_docs.add(reference)

    irds_id = load_hf_dataset_config_or_none(corpus_directory / "README.md", ["ir_dataset"])["ir_dataset"]["ir_datasets_id"]
    ds = ir_datasets.load(irds_id)
    docs_store = ds.docs_store()
    irds_loader = IrDatasetsLoader()
    irds_loader.load_dataset_for_fullrank(
        irds_id,
        corpus_directory,
        output_dataset_truth_path=None,
        skip_qrels=True,
        skip_documents=True
    )

    print(f"I export {len(all_docs)} documents that are referenced.")
    with gzip.open(corpus_directory / "corpus.jsonl.gz", "wt") as f:
        for doc_id in tqdm(sorted(list(all_docs)), "Persist documents"):
            doc = docs_store.get(doc_id)
            f.write(irds_loader.map_doc(doc, False) + "\n")

    return 0