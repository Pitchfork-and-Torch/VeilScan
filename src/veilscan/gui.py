"""Optional local Gradio UI. Not a background service."""

from __future__ import annotations


def launch(share: bool = False) -> None:
    import gradio as gr
    import numpy as np

    from veilscan.api import analyze
    from veilscan.viz import overlay

    def run(img):
        if img is None:
            return "no image", None
        rgb = np.asarray(img).astype(np.uint8)
        if rgb.ndim == 2:
            rgb = np.stack([rgb, rgb, rgb], axis=-1)
        rgb = rgb[..., :3]
        result = analyze(rgb)
        vis = overlay(rgb, result.heatmap) if result.heatmap is not None else rgb
        lines = [
            f"present={result.present}",
            f"score={result.score:.3f}",
            f"confidence={result.confidence:.3f}",
            result.explanation,
            "",
        ]
        for d in result.detectors:
            if d.skipped:
                continue
            lines.append(f"{d.detector:20} {d.score:.3f}  {d.explanation[:90]}")
        return "\n".join(lines), vis

    gr.Interface(
        fn=run,
        inputs=gr.Image(type="numpy"),
        outputs=[gr.Textbox(label="report"), gr.Image(label="heatmap")],
        title="VeilScan",
        description="Local invisible-watermark presence detector. Do not treat a score as legal proof.",
    ).launch(share=share)


if __name__ == "__main__":
    launch()
