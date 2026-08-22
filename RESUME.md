# Resume (v0.5.1)

Private GitHub: `Pitchfork-and-Torch/VeilScan` v0.5.1 doctor + camera-cover fetch.
Public lamp: https://veilscan.jonbailey.xyz/ site v1.2.0.

## Where we stopped

- `veilscan doctor` checks OP + ckpt sha256. No network.
- `scripts/fetch_camera_covers.py` + `configs/camera_covers.manifest.json` (BSDS500 test-64).
- Scan JSON may include `camera` if `configs/operating_point.camera.json` exists.
- Generator lock unchanged. Camera FPR not measured until fetch + bench.

## Next (v0.6.0)

Fetch BSDS500 (`--allow-empty-hash` once, then pin sha256). Run
`veilscan bench --covers data/covers/camera --n 50`. Write RESULTS_v0.6.
Do not overwrite generator OP. Do not flip specialist-OR.

## Do not

- Mix LSB into the FSNet cook
- Personal photos in git
- UniFreq in git
- Remover
- Gradio from a Grok Build command
