from fishweight.data import load_manifest
from fishweight.splits import PROTOCOLS, build_splits, load_splits

M = load_manifest()
ALL = set(M["img"])
S = build_splits(M)


def test_no_fold_shares_images_between_train_and_test():
    for protocol in PROTOCOLS:
        for fold in S[protocol]:
            assert not set(fold["train"]) & set(fold["test"])
            assert set(fold["train"]) | set(fold["test"]) == ALL


def test_p1_is_the_monograph_test_set():
    (fold,) = S["p1"]
    assert (len(fold["train"]), len(fold["test"])) == (170, 74)
    assert "IMG_20221122_175014.jpg" in fold["test"]


def test_p2_each_repeat_tests_every_fish_once():
    assert [f["name"] for f in S["p2"]] == [f"r{r}f{k}" for r in range(3) for k in range(5)]
    for r in range(3):
        tested = [img for f in S["p2"][r * 5:(r + 1) * 5] for img in f["test"]]
        assert sorted(tested) == sorted(ALL)


def test_p3_holds_out_one_session_group_per_fold():
    session_of = dict(zip(M["img"], M["session_group"]))
    assert sorted(f["name"] for f in S["p3"]) == sorted(set(session_of.values()))
    for fold in S["p3"]:
        assert {session_of[i] for i in fold["test"]} == {fold["name"]}
        assert fold["name"] not in {session_of[i] for i in fold["train"]}


def test_committed_files_match_a_fresh_build():
    for protocol in PROTOCOLS:
        assert load_splits(protocol) == S[protocol]
