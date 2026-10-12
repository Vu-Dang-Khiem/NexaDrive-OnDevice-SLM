"""FLEURS tiếng Việt phần TEST (người thật, câu thường) — bộ kiểm "QUÊN" cho fine-tune STT (PLAN.md
"FINE-TUNE STT"): bản fine-tune không được nghe câu thường kém hơn bản gốc. 300 câu không có chữ số
(FLEURS ghi số bằng chữ số), xáo seed 2034. Ra data/stt/fleurs_test/ theo định dạng eval.stt.chay_stt;
chấm WER bằng `--wer`.

    conda run -n vdk1 python -m eval.stt.ft.fleurs_test
    conda run -n vdk1 python -m eval.stt.ft.fleurs_test --wer runs/stt/ft/fleurs_<ban>.jsonl [...]
"""

from __future__ import annotations

import csv
import json
import random
import re
import statistics
import sys
import tarfile
from pathlib import Path

RA = Path("data/stt/fleurs_test")


def tao() -> None:
    if (RA / "manifest.jsonl").exists():
        raise SystemExit("đã có — bộ đã khoá")
    nhan = {}
    with open("data/wake/fleurs/test.tsv", encoding="utf-8") as f:
        for d in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(d) > 2 and not re.search(r"\d", d[2]):
                nhan.setdefault(d[1], d[2].strip())
    chon = sorted(nhan)
    random.Random(2034).shuffle(chon)
    chon = set(chon[:300])
    RA.mkdir(parents=True, exist_ok=True)
    with (
        tarfile.open("data/wake/fleurs/test.tar.gz") as t,
        (RA / "manifest.jsonl").open("w", encoding="utf-8") as f,
    ):
        for m in t:
            ten = Path(m.name).name
            if m.isfile() and ten in chon:
                (RA / ten).write_bytes(t.extractfile(m).read())
                f.write(
                    json.dumps(
                        {"id": Path(ten).stem, "tep": ten, "cau": nhan[ten], "prompt": "", "completion": ""},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    print("ghi", len(chon), "câu ->", RA)


def wer(tep: list[str]) -> None:
    from eval.stt.cham_chuoi import wer_cer

    for p in tep:
        rows = [json.loads(x) for x in open(p, encoding="utf-8")]
        w = [wer_cer(r["cau"], r["chu_stt"]) for r in rows]
        print(
            f"{Path(p).stem}: WER {100 * statistics.mean(a for a, _ in w):.1f}% · CER "
            f"{100 * statistics.mean(b for _, b in w):.1f}% ({len(rows)} câu)"
        )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if "--wer" in sys.argv:
        wer(sys.argv[sys.argv.index("--wer") + 1 :])
    else:
        tao()
