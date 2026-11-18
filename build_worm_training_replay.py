"""Build an offline training replay with the original viewer's canvas renderer."""

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def build(html, replay, output):
    original = html.read_text(encoding="utf-8-sig")
    start = original.index("function drawThorn(")
    end = original.index("  const hints = lv.hints", start)
    renderer = original[start:end].replace(
        "function renderLevel() {\n  const lv = getLevels()[curIdx];",
        "function drawBoard(lv) {",
    )
    if "function drawBoard(lv)" not in renderer:
        raise ValueError("Could not find the original canvas renderer")
    template = (ROOT / "worm_replay_template.html").read_text(encoding="utf-8")
    # Treat the log as data even when opened as an entirely offline HTML file.
    payload = replay.read_text(encoding="utf-8").replace("<", r"\u003c")
    output.write_text(template.replace("__BOARD_RENDERER__", renderer + "\n}")
                      .replace("__RUN_DATA__", payload), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=ROOT / "hungry_worm_map_viewer.html")
    parser.add_argument("--replay", type=Path, default=ROOT / "worm_run" / "replay.json")
    parser.add_argument("--output", type=Path, default=ROOT / "worm_training_replay.html")
    args = parser.parse_args()
    build(args.html, args.replay, args.output)
