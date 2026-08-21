"""Typer CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from veilscan import __version__
from veilscan.api import analyze_path, list_detectors
from veilscan.config import VeilConfig
from veilscan.image_io import load_rgb, save_rgb
from veilscan.viz import save_overlay

app = typer.Typer(no_args_is_help=True, add_completion=False, help="VeilScan: invisible watermark presence detector.")
console = Console()


def _cfg(config: Optional[Path], tier: Optional[str], threshold: Optional[float]) -> VeilConfig:
    cfg = VeilConfig.load(config)
    if tier:
        cfg.tier = tier
    if threshold is not None:
        cfg.threshold = threshold
    return cfg


@app.callback()
def _root() -> None:
    """VeilScan CLI."""


@app.command()
def version() -> None:
    """Print version."""
    console.print(__version__)


@app.command("list-detectors")
def list_detectors_cmd() -> None:
    """Show registered detectors."""
    table = Table(title="VeilScan detectors")
    table.add_column("name")
    table.add_column("tier")
    for d in list_detectors():
        table.add_row(d["name"], d["tier"])
    console.print(table)


@app.command()
def scan(
    path: Path = typer.Argument(..., exists=True, readable=True),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable report"),
    heatmap: Optional[Path] = typer.Option(None, help="Write overlay PNG here"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    tier: Optional[str] = typer.Option(None, help="fast|frequency|all"),
    threshold: Optional[float] = typer.Option(None),
    detectors: Optional[str] = typer.Option(None, help="Comma-separated detector names"),
    reference_dir: Optional[Path] = typer.Option(None, help="Clean reference images for WMD"),
) -> None:
    """Analyze one image (or refuse directories; use batch)."""
    if path.is_dir():
        raise typer.BadParameter("path is a directory; use `veilscan batch`")
    cfg = _cfg(config, tier, threshold)
    names = [x.strip() for x in detectors.split(",")] if detectors else None
    result = analyze_path(path, config=cfg, detectors=names, reference_dir=reference_dir)
    if heatmap:
        rgb = load_rgb(path)
        save_overlay(heatmap, rgb, result.heatmap)
    if json_out:
        console.print_json(data=result.to_json())
        return
    _print_result(path, result)


@app.command()
def batch(
    path: Path = typer.Argument(..., exists=True, readable=True),
    json_out: bool = typer.Option(False, "--json"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    tier: Optional[str] = typer.Option(None),
) -> None:
    """Scan every image under a folder."""
    from veilscan.image_io import iter_images

    cfg = _cfg(config, tier, None)
    rows = []
    for p in iter_images(path):
        r = analyze_path(p, config=cfg)
        rows.append({"path": str(p), **{k: r.to_json()[k] for k in ("present", "score", "confidence", "uncertainty")}})
        if not json_out:
            flag = "WM" if r.present else "clean"
            console.print(f"{flag:5} {r.score:.3f}  {p}")
    if json_out:
        console.print_json(data=rows)


@app.command()
def embed(
    inp: Path = typer.Argument(..., exists=True),
    out: Path = typer.Argument(...),
    family: str = typer.Option("lsb", "--family", "-f"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Embed a synthetic eval watermark (not a product hiding tool)."""
    from veilscan.generators import embed as do_embed

    rgb = load_rgb(inp)
    marked = do_embed(rgb, family, seed=seed)
    save_rgb(out, marked)
    console.print(f"wrote {out} family={family}")


@app.command()
def eval_folders(
    clean: Path = typer.Option(..., exists=True, file_okay=False),
    wm: Path = typer.Option(..., exists=True, file_okay=False),
) -> None:
    """AUC / TPR on two folders of images."""
    from veilscan.eval.harness import run_folder_eval

    report = run_folder_eval(clean, wm)
    console.print_json(data=report)


@app.command()
def selftest(
    n: int = typer.Option(6, help="Covers per family"),
    size: int = typer.Option(128),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Generate synthetic covers, embed several families, report AUC."""
    from veilscan.eval.harness import run_synthetic
    from veilscan.registry import all_detectors, ensure_loaded

    ensure_loaded()
    names = [d.name for d in all_detectors() if d.name != "wmd"]
    report = run_synthetic(n=n, size=size, detectors=names)
    if json_out:
        console.print_json(data=report)
        return
    table = Table(title="VeilScan selftest (synthetic)")
    table.add_column("family")
    table.add_column("AUC", justify="right")
    table.add_column("mean cover", justify="right")
    table.add_column("mean marked", justify="right")
    table.add_column("F1", justify="right")
    for fam, row in report["families"].items():
        table.add_row(
            fam,
            f"{row['auc']:.3f}",
            f"{row['mean_cover']:.3f}",
            f"{row['mean_marked']:.3f}",
            f"{row['f1']:.3f}",
        )
    console.print(table)
    ov = report["overall"]
    console.print(
        f"overall AUC={ov['auc']:.3f}  TPR@5%FPR={ov['tpr_at_fpr_5']:.3f}  F1={ov['f1']:.3f}"
    )


@app.command()
def robustness(
    family: str = typer.Option("dct"),
    n: int = typer.Option(4),
) -> None:
    """JPEG / resize / noise sweep on one synthetic family."""
    from veilscan.eval.harness import run_robustness

    report = run_robustness(n=n, family=family)
    console.print_json(data=report)


def _print_result(path: Path, result) -> None:
    flag = "[red]WATERMARK LIKELY[/red]" if result.present else "[green]NO STRONG MARK[/green]"
    console.print(f"{path}")
    console.print(f"  {flag}  score={result.score:.3f}  conf={result.confidence:.3f}  unc={result.uncertainty:.3f}")
    console.print(f"  {result.explanation}")
    table = Table(show_header=True)
    table.add_column("detector")
    table.add_column("score", justify="right")
    table.add_column("conf", justify="right")
    table.add_column("skip")
    table.add_column("note")
    for d in sorted(result.detectors, key=lambda r: (-(not r.skipped), -r.score)):
        note = d.explanation if len(d.explanation) < 80 else d.explanation[:77] + "..."
        table.add_row(d.detector, f"{d.score:.3f}", f"{d.confidence:.3f}", "yes" if d.skipped else "", note)
    console.print(table)
