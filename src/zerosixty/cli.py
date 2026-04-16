from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from zerosixty.pipeline import run_pipeline

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
    ] = Path("."),
    output_dir: Annotated[
        Path,
        typer.Option(
            file_okay=False,
            dir_okay=True,
            help="Directory for generated reports and flat files.",
        ),
    ] = Path("./outputs/latest"),
    members_csv: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="Explicit list-members CSV path.",
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
    min_shared_overlap: Annotated[
        int,
        typer.Option(
            min=1,
            help="Minimum shared retweeted originals before keeping an account pair.",
        ),
    ] = 2,
) -> None:
    """Run the deterministic coordination-analysis pipeline."""

    results, written = run_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        members_csv=members_csv,
        exporter_json=exporter_json,
        min_shared_overlap=min_shared_overlap,
    )

    stats = results.dataset_stats
    console.print(f"members_csv: {results.dataset_paths.members_csv}")
    console.print(f"exporter_json: {results.dataset_paths.exporter_json}")
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

    console.print("written files:")
    for name, path in written.items():
        console.print(f"- {name}: {path}")


if __name__ == "__main__":
    app()
