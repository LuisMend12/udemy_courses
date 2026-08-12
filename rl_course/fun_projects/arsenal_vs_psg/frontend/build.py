"""Inline the trained Q-tables into index.html so the page is self-contained.

The frontend has no server and no fetch - a strict CSP blocks both - so the
policy has to be baked into the file. Run the trainer with --export first, then
point this at the JSON it wrote:

    python ../football_rl.py --episodes 12000 --seed 42 --export policy.json
    python build.py policy.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
PAGE = HERE / "index.html"
NEEDLE = re.compile(r"const POLICY = .*?;\n", re.DOTALL)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)

    policy = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    blob = json.dumps(policy, ensure_ascii=False, separators=(",", ":"))
    html = PAGE.read_text(encoding="utf-8")

    if not NEEDLE.search(html):
        sys.exit("could not find the POLICY declaration in index.html")

    html = NEEDLE.sub(f"const POLICY = {blob};\n", html, count=1)
    PAGE.write_text(html, encoding="utf-8")

    kb = len(blob) / 1024
    print(f"Inlined {kb:.1f} KB of trained policy "
          f"(seed {policy['meta']['seed']}, "
          f"{policy['meta']['episodes']:,} episodes) into {PAGE.name}")


if __name__ == "__main__":
    main()
