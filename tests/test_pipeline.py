"""End-to-end on a tiny synthetic corpus: parsing robustness, eligibility, cleaning, split, features, determinism."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kl.attacks.a1_gaussian import A1Gaussian
from kl.features import FEATURE_NAMES, FeatureParams, extract_features_frame
from kl.io_aalto import eligibility, list_participant_files, load_metadata, parse_chunk, parse_keystroke_file
from kl.sessions import CleaningParams, clean_sessions, merge_counts, empty_counts, session_index
from kl.split import SPLITS, assert_disjoint, build_manifest

HEADER = "PARTICIPANT_ID\tTEST_SECTION_ID\tSENTENCE\tUSER_INPUT\tKEYSTROKE_ID\tPRESS_TIME\tRELEASE_TIME\tLETTER\tKEYCODE"
SENTENCES = ["The quick brown fox.", "Pack my box with five dozen jugs.", "Crème brûlée is nice", "How vexingly quick daft zebras jump!"]


def _write_participant(path, pid: int, n_sessions: int, seed: int, encoding="utf-8", extra_lines=()):
    rng = np.random.default_rng(seed)
    gen = A1Gaussian()
    lines = [HEADER]
    kid = 1
    for k in range(n_sessions):
        bot = gen.generate(SENTENCES[k % len(SENTENCES)], rng)  # any plausible timing works as a stand-in for humans
        for _, r in bot.iterrows():
            letter = "BKSP" if r["KEYCODE"] == 8 else r["LETTER"]
            lines.append("\t".join(map(str, [pid, 1000 + k, r["SENTENCE"], r["USER_INPUT"], kid, int(r["PRESS_TIME"]), int(r["RELEASE_TIME"]), letter, r["KEYCODE"]])))
            kid += 1
    lines += list(extra_lines)
    path.write_bytes(("\n".join(lines) + "\n").encode(encoding))


@pytest.fixture()
def corpus(tmp_path):
    raw = tmp_path / "files"
    raw.mkdir()
    meta = ["PARTICIPANT_ID\tAGE\tGENDER\tHAS_TAKEN_TYPING_COURSE\tCOUNTRY\tLAYOUT\tNATIVE_LANGUAGE\tFINGERS\tTIME_SPENT_TYPING\tKEYBOARD_TYPE\tERROR_RATE\tAVG_WPM_15\tAVG_IKI\tECPC\tKSPC\tROR"]
    kinds = ["full", "laptop", "small", "on-screen", ""]
    for pid in range(1, 41):
        kt = kinds[pid % len(kinds)]
        meta.append("\t".join([str(pid), "30", "male", "0", "US", "qwerty" if pid % 7 else "azerty", "en", "1-2", "8", kt, "0.1", "60", "150", "0.05", "1.1", "0.2"]))
        # participant 39 has no keystroke file; 38 is written latin-1 with a garbage line
        if pid == 39:
            continue
        enc = "latin-1" if pid == 38 else "utf-8"
        extra = ["garbage line without tabs", "\t".join([str(pid)] + ["x"] * 14)] if pid == 38 else []  # 1 field (padded) and 15 fields (skipped)
        _write_participant(raw / f"{pid}_keystrokes.txt", pid, 3, seed=pid, encoding=enc, extra_lines=extra)
    meta.append("41\t30\tmale\t0\tUS\tqwerty\ten\t1-2\t8\tfull\t0.1\t60\t150\t0.05\t1.1\t0.2")  # metadata row without file
    meta.append("5\t30\tmale\t0\tUS\tqwerty\ten\t1-2\t8\tfull\t0.1\t60\t150\t0.05\t1.1\t0.2")   # duplicate id
    (raw / "metadata_participants.txt").write_text("\n".join(meta) + "\n", encoding="utf-8")
    (raw / "readme.txt").write_text("readme", encoding="utf-8")
    (raw / "bad_keystrokes.txt").write_text("nonsense", encoding="utf-8")  # non-numeric id -> ignored by discovery
    return raw


def test_discovery_metadata_and_eligibility(corpus):
    files = list_participant_files(corpus)
    assert 39 not in files and 38 in files and "bad" not in files and len(files) == 39
    meta = load_metadata(corpus / "metadata_participants.txt")
    assert meta.attrs["rows_duplicate_id"] == 1 and len(meta) == 41
    elig, rep = eligibility(meta, files.keys(), ["full", "laptop"], ["small", "on-screen"])
    assert set(elig["PARTICIPANT_ID"]).isdisjoint({p for p in range(1, 41) if p % 5 in (2, 3, 4)})
    assert 39 not in set(elig["PARTICIPANT_ID"])  # eligible keyboard but no keystroke file
    assert rep["keyboard_type_counts"]["<missing>"] == 8 and rep["unmapped_values"] == {"<missing>": 8}
    assert rep["eligible"] == 17 + 0 and rep["eligible_with_file"] == len(elig)  # 16 full/laptop with files + participant 41 without
    assert 41 not in set(elig["PARTICIPANT_ID"])


def test_parse_is_robust_latin1_and_bad_lines(corpus):
    df, st = parse_keystroke_file(corpus / "38_keystrokes.txt", 38)
    assert st.encoding == "latin-1" and st.error == "" and st.lines_not_parsed >= 1
    assert df is not None and df["SENTENCE"].str.contains("Crème").any() and (df["PARTICIPANT_ID"] == 38).all()
    assert st.rows_bad_ids == 1  # the one-field garbage line is padded by pandas but its PARTICIPANT_ID is not numeric
    # wrong participant id in the file name -> rows dropped and counted, never raises
    df2, st2 = parse_keystroke_file(corpus / "1_keystrokes.txt", 999)
    assert df2 is not None and len(df2) == 0 and st2.rows_bad_ids == st2.rows_read > 0
    # missing header columns -> reported, not raised
    (corpus / "77_keystrokes.txt").write_text("A\tB\n1\t2\n", encoding="utf-8")
    df3, st3 = parse_keystroke_file(corpus / "77_keystrokes.txt", 77)
    assert df3 is None and "missing columns" in st3.error
    df4, st4 = parse_keystroke_file(corpus / "does_not_exist.txt", 5)
    assert df4 is None and st4.error


def _run_pipeline(corpus, seed: int):
    files = list_participant_files(corpus)
    meta = load_metadata(corpus / "metadata_participants.txt")
    elig, _ = eligibility(meta, files.keys(), ["full", "laptop"], ["small", "on-screen"])
    ids = np.sort(elig["PARTICIPANT_ID"].to_numpy())
    sampled = np.sort(np.random.default_rng(seed).choice(ids, size=12, replace=False))
    layout = dict(zip(elig["PARTICIPANT_ID"], elig["LAYOUT"]))
    table, stats = parse_chunk([(int(i), str(files[int(i)]), layout[i]) for i in sampled])
    kept, counts = clean_sessions(table.to_pandas(), CleaningParams())
    index = session_index(kept)
    manifest = build_manifest(index, {"train": 0.5, "val": 0.2, "test": 0.2, "attacker_pool": 0.1}, 10, seed)
    feats = extract_features_frame(kept, FeatureParams())
    return sampled, kept, counts, index, manifest, feats


def test_cleaning_rules_are_counted_separately():
    def sess(pid, tsid, n=12, sentence="abc", user_input="abc", hold=100, gap=150):
        press = np.arange(n) * (hold + gap)
        return pd.DataFrame({"PARTICIPANT_ID": pid, "TEST_SECTION_ID": tsid, "SENTENCE": sentence, "USER_INPUT": user_input,
                             "KEYSTROKE_ID": np.arange(n), "PRESS_TIME": press.astype(float), "RELEASE_TIME": (press + hold).astype(float),
                             "LETTER": "a", "KEYCODE": 65, "LAYOUT": "qwerty"})
    good = sess(1, 1)
    short = sess(1, 2, n=9)
    neg_hold = sess(1, 3, hold=-5)
    long_hold = sess(1, 4, hold=2500)
    big_gap = sess(1, 5, gap=6000)
    empty = sess(1, 6, user_input="  ")
    nan_t = sess(1, 7)
    nan_t.loc[3, "PRESS_TIME"] = np.nan
    both = sess(1, 8, n=5, hold=3000)  # too few AND hold too long -> counted in both
    df = pd.concat([good, short, neg_hold, long_hold, big_gap, empty, nan_t, both], ignore_index=True)
    kept, c = clean_sessions(df, CleaningParams())
    assert c["sessions_total"] == 8 and c["sessions_kept"] == 1 and c["dropped_total"] == 7
    assert c["drop_too_few_keystrokes"] == 2 and c["drop_bad_times"] == 1 and c["drop_empty_text"] == 1
    assert c["drop_hold_out_of_range"] == 3 and c["drop_gap_too_long"] == 1
    assert list(kept["session_id"].unique()) == ["1_1"] and len(kept) == 12
    assert merge_counts(c, c)["sessions_total"] == 16 and set(empty_counts()) <= set(c)


def test_pipeline_determinism_and_no_leakage(corpus):
    a = _run_pipeline(corpus, 42)
    b = _run_pipeline(corpus, 42)
    assert np.array_equal(a[0], b[0]) and a[4] == b[4]
    pd.testing.assert_frame_equal(a[5], b[5])
    assert not np.array_equal(a[0], _run_pipeline(corpus, 43)[0])
    _, kept, counts, index, manifest, feats = a
    assert_disjoint(manifest["users"])
    assert list(feats.columns)[:3] == ["session_id", "user_id", "LAYOUT"] and set(FEATURE_NAMES) <= set(feats.columns)
    assert np.isfinite(feats[FEATURE_NAMES].to_numpy()).all() and len(feats) == counts["sessions_kept"] == len(index)
    assert set(feats["LAYOUT"]) <= {"qwerty", "azerty"} and counts["sessions_kept"] > 0
    listed = {s for s in SPLITS for s in manifest["sessions"][s]}
    assert len(listed) <= len(index)
