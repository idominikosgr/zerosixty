from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from zerosixty.curation import build_clean_batches, resolve_clean_batch
from zerosixty.pipeline import run_pipeline, run_pipeline_clean

app = typer.Typer(
    help="Analyze X/Twitter list exports for coordination patterns.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main() -> None:
    """CLI entrypoint."""


@app.command()
def analyze(
    input_dir: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            help="Directory containing the raw export files.",
        ),
    ] = Path(),
    output_dir: Annotated[
        Path,
        typer.Option(
            file_okay=False,
            dir_okay=True,
            help="Directory for generated reports and flat files.",
        ),
    ] = Path("./outputs/latest"),
    members_file: Annotated[
        Path | None,
        typer.Option(
            "--members-file",
            "--members-csv",
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="Explicit list-members export path. Supports CSV and JSON.",
        ),
    ] = None,
    exporter_json: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="Explicit twitter-web-exporter JSON path.",
        ),
    ] = None,
    extra_members_file: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="Optional text, JSON, or CSV file with extra member handles or profile URLs.",
        ),
    ] = None,
    min_shared_overlap: Annotated[
        int,
        typer.Option(
            min=1,
            help="Minimum shared retweeted originals before keeping an account pair.",
        ),
    ] = 2,
    enable_ml: Annotated[
        bool,
        typer.Option(
            "--with-ml/--no-ml",
            help="Run the unsupervised ML baseline alongside deterministic analysis.",
        ),
    ] = True,
    ml_clusters: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Explicit cluster count for the ML baseline. Defaults to silhouette selection.",
        ),
    ] = None,
    ml_random_state: Annotated[
        int,
        typer.Option(
            help="Random seed for the ML baseline.",
        ),
    ] = 42,
) -> None:
    """Run the deterministic coordination-analysis pipeline and ML baseline."""

    results, written = run_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        members_file=members_file,
        exporter_json=exporter_json,
        extra_members_file=extra_members_file,
        min_shared_overlap=min_shared_overlap,
        enable_ml=enable_ml,
        ml_clusters=ml_clusters,
        ml_random_state=ml_random_state,
    )

    stats = results.dataset_stats
    console.print(f"members_file: {results.dataset_paths.members_file}")
    console.print(f"exporter_json: {results.dataset_paths.exporter_json}")
    if results.dataset_paths.extra_members_file is not None:
        console.print(f"extra_members_file: {results.dataset_paths.extra_members_file}")
    console.print(f"output_dir: {output_dir.resolve()}")
    console.print()
    console.print(
        "dataset: "
        f"{stats.member_count} members, {stats.tweet_count} tweets, "
        f"{stats.active_account_count} active accounts, {stats.retweet_count} retweets"
    )

    table = Table(title="Top coordination indicators")
    table.add_column("account")
    table.add_column("score", justify="right")
    table.add_column("tweets", justify="right")
    table.add_column("retweet_ratio", justify="right")
    table.add_column("first_retweeter_count", justify="right")
    table.add_column("top_amplified")
    for account in results.account_summaries[:10]:
        table.add_row(
            account.account_handle,
            f"{account.coordination_score:.2f}",
            str(account.tweet_count),
            f"{account.retweet_ratio:.2f}",
            str(account.first_retweeter_count),
            account.top_amplified_account or "-",
        )
    console.print(table)

    ml_summary = results.ml_run_summary
    console.print(
        "ml: "
        f"status={ml_summary.status} "
        f"samples={ml_summary.sample_count} "
        f"features={ml_summary.input_feature_count} "
        f"clusters={ml_summary.cluster_count}"
    )
    if results.ml_account_summaries:
        top_anomaly = results.ml_account_summaries[0]
        console.print(
            "top_ml_anomaly: "
            f"{top_anomaly.account_handle} "
            f"cluster={top_anomaly.cluster_id} "
            f"score={top_anomaly.anomaly_score:.4f}"
        )

    console.print("written files:")
    for name, path in written.items():
        console.print(f"- {name}: {path}")


@app.command("build-clean")
def build_clean(
    raw_dir: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            help="Directory containing immutable raw snapshots.",
        ),
    ] = Path("./datasets-raw"),
    clean_dir: Annotated[
        Path,
        typer.Option(
            file_okay=False,
            dir_okay=True,
            help="Directory where versioned clean batches are written.",
        ),
    ] = Path("./datasets-clean"),
    extra_members_file: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="Optional explicit manual-member file. Defaults to auto-discovery.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force/--no-force",
            help="Rebuild batches even when source fingerprints are unchanged.",
        ),
    ] = False,
) -> None:
    """Build versioned clean datasets from raw snapshots."""

    results, index_path = build_clean_batches(
        raw_dir=raw_dir,
        clean_dir=clean_dir,
        extra_members_file=extra_members_file,
        force=force,
    )

    table = Table(title="Clean dataset batches")
    table.add_column("batch")
    table.add_column("status")
    table.add_column("members", justify="right")
    table.add_column("tweets", justify="right")
    table.add_column("retweets", justify="right")
    table.add_column("exporter")
    table.add_column("members file")
    for item in results:
        table.add_row(
            item.batch_id,
            item.status,
            str(item.member_count),
            str(item.tweet_count),
            str(item.retweet_count),
            item.exporter_file.name,
            item.members_file.name,
        )

    console.print(table)
    console.print(
        "summary: "
        f"total={len(results)} "
        f"built={sum(1 for item in results if item.status == 'built')} "
        f"skipped={sum(1 for item in results if item.status == 'skipped')}"
    )
    console.print(f"index: {index_path.resolve()}")


@app.command("analyze-clean")
def analyze_clean(
    clean_dir: Annotated[
        Path,
        typer.Option(
            file_okay=False,
            dir_okay=True,
            help="Directory containing versioned clean batches (`datasets-clean`).",
        ),
    ] = Path("./datasets-clean"),
    output_dir: Annotated[
        Path,
        typer.Option(
            file_okay=False,
            dir_okay=True,
            help="Root directory for analysis outputs generated from clean batches.",
        ),
    ] = Path("./outputs/clean"),
    batch_id: Annotated[
        str | None,
        typer.Option(
            help="Batch id from datasets-clean/index.json. Defaults to latest batch.",
        ),
    ] = None,
    auto_build: Annotated[
        bool,
        typer.Option(
            "--auto-build/--no-auto-build",
            help="Refresh clean batches from raw snapshots before analysis.",
        ),
    ] = True,
    raw_dir: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            help="Raw snapshot directory used only when auto-build is enabled.",
        ),
    ] = Path("./datasets-raw"),
    extra_members_file: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="Optional explicit manual-member file used during auto-build.",
        ),
    ] = None,
    force_clean: Annotated[
        bool,
        typer.Option(
            "--force-clean/--no-force-clean",
            help="Force clean-batch rebuild when auto-build runs.",
        ),
    ] = False,
    batch_subdir: Annotated[
        bool,
        typer.Option(
            "--batch-subdir/--no-batch-subdir",
            help="Write outputs under output_dir/<batch_id>.",
        ),
    ] = True,
    min_shared_overlap: Annotated[
        int,
        typer.Option(
            min=1,
            help="Minimum shared retweeted originals before keeping an account pair.",
        ),
    ] = 2,
    enable_ml: Annotated[
        bool,
        typer.Option(
            "--with-ml/--no-ml",
            help="Run the unsupervised ML baseline alongside deterministic analysis.",
        ),
    ] = True,
    ml_clusters: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Explicit cluster count for the ML baseline. Defaults to silhouette selection.",
        ),
    ] = None,
    ml_random_state: Annotated[
        int,
        typer.Option(
            help="Random seed for the ML baseline.",
        ),
    ] = 42,
) -> None:
    """Run analysis directly from one clean batch."""

    if auto_build:
        build_results, index_path = build_clean_batches(
            raw_dir=raw_dir,
            clean_dir=clean_dir,
            extra_members_file=extra_members_file,
            force=force_clean,
        )
        console.print(
            "clean refresh: "
            f"total={len(build_results)} "
            f"built={sum(1 for item in build_results if item.status == 'built')} "
            f"skipped={sum(1 for item in build_results if item.status == 'skipped')}"
        )
        console.print(f"clean index: {index_path.resolve()}")

    selected_batch = resolve_clean_batch(clean_dir=clean_dir, batch_id=batch_id)
    selected_output_dir = (
        output_dir / selected_batch.batch_id
        if batch_subdir
        else output_dir
    )
    results, written = run_pipeline_clean(
        members_csv=selected_batch.members_csv,
        tweets_csv=selected_batch.tweets_csv,
        output_dir=selected_output_dir,
        source_members_file=selected_batch.source_members_file,
        source_exporter_file=selected_batch.source_exporter_file,
        source_extra_members_file=selected_batch.source_extra_members_file,
        min_shared_overlap=min_shared_overlap,
        enable_ml=enable_ml,
        ml_clusters=ml_clusters,
        ml_random_state=ml_random_state,
    )

    stats = results.dataset_stats
    console.print(f"batch_id: {selected_batch.batch_id}")
    console.print(f"clean_members_csv: {selected_batch.members_csv}")
    console.print(f"clean_tweets_csv: {selected_batch.tweets_csv}")
    if selected_batch.source_members_file is not None:
        console.print(f"source_members_file: {selected_batch.source_members_file}")
    if selected_batch.source_exporter_file is not None:
        console.print(f"source_exporter_json: {selected_batch.source_exporter_file}")
    if selected_batch.source_extra_members_file is not None:
        console.print(f"source_extra_members_file: {selected_batch.source_extra_members_file}")
    console.print(f"output_dir: {selected_output_dir.resolve()}")
    console.print()
    console.print(
        "dataset: "
        f"{stats.member_count} members, {stats.tweet_count} tweets, "
        f"{stats.active_account_count} active accounts, {stats.retweet_count} retweets"
    )

    table = Table(title="Top coordination indicators")
    table.add_column("account")
    table.add_column("score", justify="right")
    table.add_column("tweets", justify="right")
    table.add_column("retweet_ratio", justify="right")
    table.add_column("first_retweeter_count", justify="right")
    table.add_column("top_amplified")
    for account in results.account_summaries[:10]:
        table.add_row(
            account.account_handle,
            f"{account.coordination_score:.2f}",
            str(account.tweet_count),
            f"{account.retweet_ratio:.2f}",
            str(account.first_retweeter_count),
            account.top_amplified_account or "-",
        )
    console.print(table)

    ml_summary = results.ml_run_summary
    console.print(
        "ml: "
        f"status={ml_summary.status} "
        f"samples={ml_summary.sample_count} "
        f"features={ml_summary.input_feature_count} "
        f"clusters={ml_summary.cluster_count}"
    )
    if results.ml_account_summaries:
        top_anomaly = results.ml_account_summaries[0]
        console.print(
            "top_ml_anomaly: "
            f"{top_anomaly.account_handle} "
            f"cluster={top_anomaly.cluster_id} "
            f"score={top_anomaly.anomaly_score:.4f}"
        )

    console.print("written files:")
    for name, path in written.items():
        console.print(f"- {name}: {path}")


if __name__ == "__main__":
    app()
