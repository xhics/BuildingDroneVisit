from __future__ import annotations

from hotel_pipeline.conditioning.vertical_registration import assess_metrics


def test_vertical_registration_gate_accepts_only_decisive_holdout() -> None:
    status, reasons = assess_metrics(
        fit_points=160,
        holdout_points=40,
        holdout_support_fraction_1m=0.72,
        holdout_median_m=0.42,
        holdout_p90_m=1.35,
        best_negative_support_fraction_1m=0.40,
    )

    assert status == "accepted"
    assert reasons == []


def test_vertical_registration_gate_refuses_ambiguous_hypothesis() -> None:
    status, reasons = assess_metrics(
        fit_points=160,
        holdout_points=40,
        holdout_support_fraction_1m=0.41,
        holdout_median_m=1.2,
        holdout_p90_m=3.1,
        best_negative_support_fraction_1m=0.38,
    )

    assert status == "refused"
    assert any("support within 1 m" in reason for reason in reasons)
    assert any("negative-control margin" in reason for reason in reasons)
