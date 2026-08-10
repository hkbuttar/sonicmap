from results.compile_results import compile_results, render_report


def test_compiled_results_cover_all_core_sections():
    summary = compile_results(".")
    assert {"augmentation", "genre_classification", "mood_regression", "embedding", "similarity", "generalization", "playlist"}.issubset(summary["section"])
    assert summary["value"].notna().all()


def test_report_states_negative_triplet_result_honestly():
    report = render_report(compile_results("."))
    assert "did **not** beat" in report
    assert "degraded less" in report
    assert "promising rather than conclusive" in report
    assert "metadata baseline is an oracle-like ceiling" in report
