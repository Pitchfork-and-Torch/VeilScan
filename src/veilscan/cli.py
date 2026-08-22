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

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="VeilScan: invisible watermark presence detector and keyless plaintext decoder.",
)
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


@app.command()
def doctor() -> None:
    """Check local stack: OP file, checkpoint hashes, torch/cuda. No network."""
    from veilscan.doctor import run_doctor

    report = run_doctor()
    console.print_json(data=report)
    if not report.get("ok"):
        raise typer.Exit(code=2)


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
    _print_result(path, result, peak_ok=set(cfg.peak_ok))


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
def decode(
    path: Path = typer.Argument(..., exists=True, readable=True),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable report"),
    out: Optional[Path] = typer.Option(None, "--out", help="HUD + dossier PNG path"),
    no_report: bool = typer.Option(False, "--no-report", help="Skip the PNG artifact"),
) -> None:
    """Recover keyless plaintext and write a HUD + dossier PNG by default."""
    from veilscan.decode import decode_path as do_decode
    from veilscan.decode.container import load_raw_rgb
    from veilscan.decode.report import default_report_path, render_decode_report

    result = do_decode(path)
    if json_out:
        console.print_json(data=result.to_json())
    elif result.found:
        console.print(f"{path}")
        console.print(f"  FOUND  family={result.family}  layout={result.layout}  conf={result.confidence:.3f}")
        console.print(result.text or "")
        for hs in result.hotspots[:8]:
            console.print(f"  hotspot x={hs.x} y={hs.y} {hs.w}x{hs.h} corr={hs.corr:.3f}")
        for note in result.notes:
            console.print(f"  note: {note}")
    else:
        console.print(f"{path}")
        console.print("  NO KEYLESS PLAINTEXT")
        for hs in result.hotspots[:8]:
            console.print(f"  hotspot x={hs.x} y={hs.y} {hs.w}x{hs.h} corr={hs.corr:.3f}")
        for note in result.notes:
            console.print(f"  note: {note}")
    if not no_report:
        rgb, _, _ = load_raw_rgb(path.read_bytes())
        if rgb is not None:
            dest = out or default_report_path(path)
            render_decode_report(rgb, result, path, dest)
            console.print(f"  report {dest}")


@app.command()
def inspect(
    path: Path = typer.Argument(..., exists=True, readable=True),
    json_out: bool = typer.Option(False, "--json", help="Scan + decode as one JSON object"),
    out: Optional[Path] = typer.Option(None, "--out", help="HUD + dossier PNG path"),
    no_report: bool = typer.Option(False, "--no-report", help="Skip the PNG artifact"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    threshold: Optional[float] = typer.Option(None),
) -> None:
    """Scan presence then decode keyless text. One JSON, one HUD stamp."""
    from veilscan.decode import decode_path as do_decode
    from veilscan.decode.container import load_raw_rgb
    from veilscan.decode.report import default_report_path, render_decode_report
    from veilscan.types import SCHEMA_VERSION

    cfg = _cfg(config, None, threshold)
    scan_res = analyze_path(path, config=cfg)
    dec = do_decode(path)
    if json_out:
        console.print_json(
            data={
                "schema_version": SCHEMA_VERSION,
                "path": str(path),
                "scan": scan_res.to_json(),
                "decode": dec.to_json(),
            }
        )
    else:
        _print_result(path, scan_res, peak_ok=set(cfg.peak_ok))
        if dec.found:
            console.print(f"  DECODE FOUND  family={dec.family}  layout={dec.layout}  conf={dec.confidence:.3f}")
            console.print(f"  {dec.text}")
        else:
            console.print("  DECODE NO KEYLESS PLAINTEXT")
        for hs in dec.hotspots[:8]:
            console.print(f"  hotspot x={hs.x} y={hs.y} {hs.w}x{hs.h} corr={hs.corr:.3f}")
    if not no_report:
        rgb, _, _ = load_raw_rgb(path.read_bytes())
        if rgb is not None:
            dest = out or default_report_path(path)
            render_decode_report(rgb, dec, path, dest, scan=scan_res)
            console.print(f"  report {dest}")


@app.command("embed-text")
def embed_text_cmd(
    inp: Path = typer.Argument(..., exists=True, readable=True),
    out: Path = typer.Argument(...),
    message: str = typer.Option(..., "--message", "-m", help="UTF-8 plaintext to plant"),
    layout: str = typer.Option("r-bit0-msb", "--layout"),
    family: str = typer.Option("lsb", "--family", "-f", help="lsb | png-text | jpeg-com"),
) -> None:
    """Eval-only: plant keyless plaintext so decode can be tested. Not a hiding product."""
    from veilscan.decode.embed_text import embed_text_path

    path = embed_text_path(inp, out, message, layout_id=layout, family=family)
    console.print(f"wrote {path} family={family} layout={layout}")


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
    per_detector: bool = typer.Option(False, "--per-detector", help="Also print per-head AUC by family"),
) -> None:
    """Generate synthetic covers, embed several families, report AUC."""
    from veilscan.eval.harness import run_synthetic
    from veilscan.registry import all_detectors, ensure_loaded

    ensure_loaded()
    names = [d.name for d in all_detectors() if d.name != "wmd"]
    report = run_synthetic(n=n, size=size, detectors=names, per_detector=per_detector)
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
    if per_detector and report.get("detectors"):
        _print_detector_auc(report["detectors"])


@app.command()
def loao(
    n: int = typer.Option(3, help="Covers per held-out family"),
    size: int = typer.Option(128),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Leave-one-family-out report (classical path; deep train-holdout is Phase 2)."""
    from veilscan.eval.harness import run_loao
    from veilscan.registry import all_detectors, ensure_loaded

    ensure_loaded()
    names = [d.name for d in all_detectors() if d.name != "wmd"]
    report = run_loao(n=n, size=size, detectors=names)
    if json_out:
        console.print_json(data=report)
        return
    console.print(report["protocol"])
    table = Table(title="LOAO ensemble (held-out family vs matched covers)")
    table.add_column("held out")
    table.add_column("AUC", justify="right")
    table.add_column("mean cover", justify="right")
    table.add_column("mean marked", justify="right")
    for fam, row in report["held_out_ensemble"].items():
        table.add_row(fam, f"{row['auc']:.3f}", f"{row['mean_cover']:.3f}", f"{row['mean_marked']:.3f}")
    console.print(table)
    console.print(f"deep_loao={report['deep_loao']}")
    if report.get("detectors"):
        _print_detector_auc(report["detectors"])


def _print_detector_auc(heads: dict) -> None:
    table = Table(title="Per-detector AUC by family")
    fams = sorted({f for rec in heads.values() for f in rec})
    table.add_column("detector")
    for f in fams:
        table.add_column(f, justify="right")
    for name in sorted(heads):
        row = [name]
        for f in fams:
            cell = heads[name].get(f)
            row.append(f"{cell['auc']:.3f}" if cell else "-")
        table.add_row(*row)
    console.print(table)


@app.command()
def robustness(
    family: str = typer.Option("dct"),
    n: int = typer.Option(4),
    all_families: bool = typer.Option(False, "--all-families"),
    json_out: bool = typer.Option(True, "--json/--table"),
) -> None:
    """JPEG / resize / noise sweep. Retention vs identity AUC."""
    from veilscan.eval.harness import run_robustness, run_robustness_all

    if all_families:
        report = run_robustness_all(n=n)
    else:
        report = run_robustness(n=n, family=family)
    console.print_json(data=report)


@app.command()
def calibrate(
    n: int = typer.Option(6),
    size: int = typer.Option(96),
    out: Path = typer.Option(Path("configs/calibration.json")),
) -> None:
    """Fit affine-logit maps on synthetic pairs. Writes JSON (does not train nets)."""
    from veilscan.calibrate import fit_calibration, save_calibration

    data = fit_calibration(n=n, size=size)
    save_calibration(data, out)
    console.print(f"wrote {out} ensemble a={data['ensemble']['a']:.3f} b={data['ensemble']['b']:.3f}")


@app.command("wmd-scan")
def wmd_scan(
    suspects: Path = typer.Argument(..., exists=True, file_okay=False),
    reference_dir: Path = typer.Option(..., "--reference-dir", exists=True, file_okay=False),
    rounds: int = typer.Option(2),
    steps: int = typer.Option(4),
) -> None:
    """Dataset WMD prune (Pan et al. idea). Detection only. Needs a clean reference folder."""
    from veilscan.detectors.blackbox import wmd_prune_scan
    from veilscan.image_io import iter_images, load_rgb
    from veilscan.config import resolve_device, VeilConfig

    cfg = VeilConfig.load()
    sus = [load_rgb(p) for p in iter_images(suspects)[:32]]
    refs = [load_rgb(p) for p in iter_images(reference_dir)[:16]]
    report = wmd_prune_scan(sus, refs, rounds=rounds, steps=steps, device=resolve_device(cfg.device))
    console.print_json(data=report)


@app.command("export-onnx")
def export_onnx_cmd(
    out: Path = typer.Option(Path("checkpoints/residual_cnn.onnx")),
    ckpt: Path | None = typer.Option(None),
) -> None:
    """Export ResidualCNN (untrained or checkpoint) to ONNX."""
    from veilscan.export import export_residual_cnn

    path = export_residual_cnn(out, ckpt)
    console.print(f"wrote {path}")



@app.command()
def bench(
    n: int = typer.Option(None, help="Override protocol n (pytest uses 2; real cut is 20 then 50)"),
    size: int = typer.Option(None),
    protocol: Optional[Path] = typer.Option(None, "--protocol", "-p"),
    json_out: Optional[Path] = typer.Option(None, "--json-out"),
    write_operating_point: bool = typer.Option(False, "--write-operating-point"),
    attacks: Optional[str] = typer.Option(None, help="Comma list, default protocol attacks"),
    styles: Optional[str] = typer.Option(None, help="Comma list sine,photo"),
    covers: Optional[Path] = typer.Option(None, "--covers", help="Folder of real camera images. Never written into git."),
    corpus_id: Optional[str] = typer.Option(None, "--corpus-id", help="camera = BSDS sidecar. Other slugs write their own json/OP files."),
) -> None:
    """Frozen bench. Generator by default. --covers DIR is the camera adapter."""
    from veilscan.eval.bench import ROOT, load_camera_covers, load_protocol, run_bench, write_outputs

    proto = load_protocol(protocol)
    atk = [x.strip() for x in attacks.split(",")] if attacks else None
    st = [x.strip() for x in styles.split(",")] if styles else None
    plates = None
    json_path = json_out
    op_path = None
    slug = (corpus_id or "").strip() or None
    if covers:
        want_n = int(n if n is not None else proto.get("n", 20))
        want_size = int(size if size is not None else proto.get("size", 128))
        plates = load_camera_covers(covers, want_n, want_size, seed=int(proto.get("seed", 20260822)))
        slug = slug or "camera"
        if json_path is None:
            json_path = ROOT / "docs" / "bench" / f"{slug}.json"
        if write_operating_point:
            frozen = (ROOT / "data" / "covers" / "camera").resolve()
            if slug == "camera":
                if covers.resolve() != frozen:
                    console.print("ERROR: refusing to overwrite the BSDS camera OP from a different --covers folder. Pass --corpus-id.")
                    raise typer.Exit(code=2)
                op_path = ROOT / "configs" / "operating_point.camera.json"
            else:
                op_path = ROOT / "configs" / f"operating_point.{slug}.json"
    report = run_bench(proto, n=n, size=size, attacks=atk, styles=st, covers=plates, corpus_id=slug)
    written = write_outputs(
        report,
        json_path=json_path,
        md_path=(json_path.with_suffix(".md") if json_path is not None else None),
        operating_point_path=op_path,
        write_operating_point=bool(write_operating_point),
    )
    op = report.get("operating_point") or {}
    ab = report.get("ab") or {}
    console.print(f"bench n={report['n']} size={report['size']} corpus={report.get('corpus')} corpus_id={report.get('corpus_id')} protocol={report.get('protocol_id')}")
    console.print(f"operating_point id={op.get('id')} status={op.get('status')} threshold={op.get('threshold')} fusion={op.get('fusion_mode')}")
    locks = (report.get("fpr_at_locks") or {}).get("locks") or {}
    if locks:
        bits = [f"{k}={v.get('fpr')}@{v.get('threshold')}" for k, v in locks.items()]
        console.print("fpr_at_locks " + " ".join(bits))
    nested = ab.get("nested_holdout") or {}
    if nested:
        console.print(f"nested_holdout legacy_fpr={nested.get('legacy_fpr')} or_fpr={nested.get('or_fpr')} flip={nested.get('flip_default')}")
    for k, v in written.items():
        console.print(f"  wrote {k} {v}")


def _print_result(path: Path, result, peak_ok: set[str] | None = None) -> None:
    flag = "[red]WATERMARK LIKELY[/red]" if result.present else "[green]NO STRONG MARK[/green]"
    console.print(f"{path}")
    console.print(f"  {flag}  score={result.score:.3f}  conf={result.confidence:.3f}  unc={result.uncertainty:.3f}")
    hint = getattr(result, "family_hint", "none")
    console.print(
        f"  family={hint}  lsb={getattr(result, 'lsb_score', 0):.3f}  "
        f"freq={getattr(result, 'freq_score', 0):.3f}  class={getattr(result, 'class_score', 0):.3f}"
    )
    console.print(f"  {result.explanation}")
    table = Table(show_header=True)
    table.add_column("detector")
    table.add_column("score", justify="right")
    table.add_column("conf", justify="right")
    table.add_column("skip")
    table.add_column("core")
    table.add_column("note")
    allow = peak_ok or set()

    def _key(r):
        core = r.detector in allow
        return (r.skipped, not core, -r.score)

    for d in sorted(result.detectors, key=_key):
        note = d.explanation if len(d.explanation) < 80 else d.explanation[:77] + "..."
        table.add_row(
            d.detector,
            f"{d.score:.3f}",
            f"{d.confidence:.3f}",
            "yes" if d.skipped else "",
            "yes" if d.detector in allow else "",
            note,
        )
    console.print(table)
