import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from veilscan.api import analyze
from veilscan.cli import app
from veilscan.decode.container import (
    plant_jpeg_com,
    plant_png_itxt,
    plant_png_text,
    plant_png_ztxt,
)
from veilscan.decode.embed_text import embed_text_array
from veilscan.generators import embed_dct, embed_dwt, embed_lsb, synthetic_cover
import numpy as np

runner = CliRunner()
MSG = "The veil is in the pixels."


def test_inspect_json_has_scan_and_decode(tmp_path: Path) -> None:
    cover = synthetic_cover(96, 96, np.random.default_rng(3), style="sine")
    marked = embed_text_array(cover, MSG)
    src = tmp_path / "plate.png"
    Image.fromarray(marked).save(src)
    r = runner.invoke(app, ["inspect", str(src), "--json", "--no-report"])
    assert r.exit_code == 0, r.output
    payload = json.loads(r.stdout)
    assert "scan" in payload
    assert "decode" in payload
    assert "jpeg" in payload
    assert "heads" in payload
    assert payload["decode"]["found"] is True
    assert payload["decode"]["text"] == MSG
    assert payload["decode"]["family"] == "lsb"
    assert payload["heads"]["chi_square"] is not None
    assert payload["heads"]["dct"] is not None
    assert payload["heads"]["dwt"] is not None
    assert payload["scan"]["family_hint"] is not None


def test_inspect_jpeg_qtable_and_comment(tmp_path: Path) -> None:
    cover = synthetic_cover(96, 96, np.random.default_rng(7), style="photo")
    src = tmp_path / "q50.jpg"
    src.write_bytes(plant_jpeg_com(cover, MSG, quality=50))
    r = runner.invoke(app, ["inspect", str(src), "--json", "--no-report"])
    assert r.exit_code == 0, r.output
    payload = json.loads(r.stdout)
    jpeg = payload["jpeg"]
    assert jpeg["container"] is True
    assert jpeg["luma_q_34"] == 51
    assert jpeg["luma_q_43"] == 56
    assert jpeg["quality_est"] == 50
    assert jpeg["dct_pair_34_43"] is not None
    assert payload["scan"]["jpeg_luma_q_34"] == 51
    assert payload["scan"]["jpeg_dct_pair_34_43"] == jpeg["dct_pair_34_43"]
    assert payload["decode"]["found"] is True
    assert payload["decode"]["family"] == "container"
    assert payload["decode"]["text"] == MSG
    assert payload["decode"]["layout"] == "jpeg-COM"
    human = runner.invoke(app, ["inspect", str(src), "--no-report"])
    assert human.exit_code == 0, human.output
    assert "luma_q(3,4)=51" in human.output
    assert "luma_q(4,3)=56" in human.output
    assert "dct_pair(3,4)+(4,3)=" in human.output
    assert MSG in human.output


def test_inspect_png_text_container(tmp_path: Path) -> None:
    cover = synthetic_cover(64, 64, np.random.default_rng(8), style="sine")
    src = tmp_path / "comment.png"
    src.write_bytes(plant_png_text(cover, MSG, key="Comment"))
    r = runner.invoke(app, ["inspect", str(src), "--json", "--no-report"])
    assert r.exit_code == 0, r.output
    payload = json.loads(r.stdout)
    assert payload["decode"]["found"] is True
    assert payload["decode"]["family"] == "container"
    assert payload["decode"]["text"] == MSG
    assert "tEXt" in (payload["decode"]["layout"] or "")
    assert payload["jpeg"]["container"] is False


def test_inspect_png_ztxt_itxt(tmp_path: Path) -> None:
    cover = synthetic_cover(64, 64, np.random.default_rng(10), style="sine")
    zsrc = tmp_path / "z.png"
    isrc = tmp_path / "i.png"
    zsrc.write_bytes(plant_png_ztxt(cover, MSG, key="Comment"))
    isrc.write_bytes(plant_png_itxt(cover, MSG, key="Description"))
    z = json.loads(runner.invoke(app, ["inspect", str(zsrc), "--json", "--no-report"]).stdout)
    i = json.loads(runner.invoke(app, ["inspect", str(isrc), "--json", "--no-report"]).stdout)
    assert z["decode"]["found"] is True
    assert z["decode"]["text"] == MSG
    assert "zTXt" in (z["decode"]["layout"] or "")
    assert i["decode"]["found"] is True
    assert i["decode"]["text"] == MSG
    assert "iTXt" in (i["decode"]["layout"] or "")


def test_lsb_dct_dwt_presence_gap() -> None:
    cover = synthetic_cover(128, 128, np.random.default_rng(9), style="sine")
    lsb = embed_lsb(cover, np.random.default_rng(10), rate=0.8)
    dct = embed_dct(cover, np.random.default_rng(11), amp=12.0)
    dwt = embed_dwt(cover, np.random.default_rng(12), strength=0.2)
    names = ["chi_square", "dct", "dwt"]
    rc = analyze(cover, detectors=names)
    rl = analyze(lsb, detectors=names)
    rd = analyze(dct, detectors=names)
    rw = analyze(dwt, detectors=names)

    def _score(result, name: str) -> float:
        return next(d.score for d in result.detectors if d.detector == name)

    assert _score(rl, "chi_square") > _score(rc, "chi_square")
    assert _score(rd, "dct") > _score(rc, "dct")
    assert _score(rw, "dwt") > _score(rc, "dwt")
    pair_c = next(d.extras["pair_34_43"] for d in rc.detectors if d.detector == "dct")
    pair_d = next(d.extras["pair_34_43"] for d in rd.detectors if d.detector == "dct")
    assert pair_d > pair_c
