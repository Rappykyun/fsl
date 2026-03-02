from pathlib import Path

import pandas as pd

from fsl.data.io import filter_word_rows, resolve_video_path


def test_filter_word_rows_keeps_only_selected_ids() -> None:
    df = pd.DataFrame(
        {
            "id_label": [0, 1, 2, 7, 50],
            "label": ["GOOD MORNING", "GOOD AFTERNOON", "GOOD EVENING", "THANK YOU", "TOMORROW"],
        }
    )

    out = filter_word_rows(df, [0, 2, 7, 8])
    assert sorted(out["id_label"].unique().tolist()) == [0, 2, 7]


def test_resolve_video_path_handles_windows_style_csv_paths(tmp_path: Path) -> None:
    root = tmp_path / "root"
    clip = root / "clips" / "clips" / "3" / "1.MOV"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.touch()

    resolved = resolve_video_path(root, "clips\\3\\1.MOV")
    assert resolved.exists()
    assert resolved == clip
