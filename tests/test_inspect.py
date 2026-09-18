from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from veilscan.cli import app
from veilscan.decode.embed_text import embed_text_array
from veilscan.generators import synthetic_cover
import numpy as np

runner = CliRunner()


def test_inspect_json_has_scan_and_decode(tmp_path: Path) -> None:
    cover = synthetic_cover(96, 96, np.random.default_rng(3), style="sine")
    marked = embed_text_array(cover, "The veil is in the pixels.")
    src = tmp_path / "plate.png"
    Image.fromarray(marked).save(src)
    r = runner.invoke(app, ["inspect", str(src), "--json", "--no-report"])
    assert r.exit_code == 0, r.output
    assert '"scan"' in r.stdout
    assert '"decode"' in r.stdout
    assert '"family_hint"' in r.stdout
    assert "The veil is in the pixels." in r.stdout
    assert '"found": true' in r.stdout
