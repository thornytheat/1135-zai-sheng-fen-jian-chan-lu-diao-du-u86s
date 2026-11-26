from datetime import date, timedelta
from types import SimpleNamespace
import pytest
from sortyield.yield_forecast.contracts import Overhaul, RunHistory
from sortyield.yield_forecast.engine import (
    calibrate,
    clipped_prediction,
    model_vector,
    predict,
    resolve,
)

DAY = date(2026, 11, 1)
RESUME = DAY + timedelta(days=12)


def history(index, line="涡电流线", material="PP"):
    handled = DAY + timedelta(days=index)
    return RunHistory(
        handled,
        handled + timedelta(days=1),
        material,
        model_vector(1000 + index * 150, 0.03 + index % 6 * 0.03, 0.02 + index % 4 * 0.03, line),
        0.72 + index % 5 * 0.02,
        line,
    )


def lot(kg=3400):
    return SimpleNamespace(lot="P1", material="PP", line="涡电流线", kg=kg)


def audit():
    return SimpleNamespace(
        approved_samples=True, impurity=0.09, moisture=0.05, upper_impurity=0.2
    )


def trained(rows):
    return calibrate(rows, DAY, DAY + timedelta(days=20), (Overhaul("涡电流线", RESUME),))


def test_post_overhaul_calibration_uses_five_unseen_samples_and_borrows_old_line_records():
    curves, fits = trained([history(index) for index in range(19)])
    fit = fits[0]
    assert (fit.post_count, fit.calibration, fit.borrowed) == (7, 5, 3)
    assert fit.notice == "检修后校准"
    curve = curves[("涡电流线", "PP")]
    assert curve.validation == 5 and curve.count == 5
    assert curve.envelope is not None


def test_calibration_samples_never_enter_the_fit_itself():
    rows = [history(index) for index in range(19)]
    curves, _ = trained(rows)
    curve = curves[("涡电流线", "PP")]
    post = rows[12:]
    calibration = post[-5:]
    training = post[:-5]
    residuals = [
        abs(row.yield_rate - clipped_prediction(row.features, curve.center, curve.scale, curve.params, curve.rate))
        for row in calibration
    ]
    assert max(residuals) == pytest.approx(curve.loss_radius)
    assert all(row not in training for row in calibration)


def test_fewer_than_five_post_overhaul_samples_blocks_the_line():
    curves, fits = trained([history(index) for index in range(16)])
    assert curves == {}
    assert fits[0].notice == "检修后未成熟样本不足"
    assert resolve(curves, "涡电流线", "PP", frozenset({"涡电流线"})) is None


def test_unclosed_post_overhaul_labels_do_not_count():
    rows = [history(index) for index in range(19)]
    late = RunHistory(
        RESUME,
        DAY + timedelta(days=40),
        "PP",
        model_vector(9999, 0.09, 0.05, "涡电流线"),
        0.10,
        "涡电流线",
    )
    curves, fits = trained([*rows, late])
    assert fits[0].post_count == 7 and curves[("涡电流线", "PP")].count == 5


def test_borrowed_records_come_only_from_the_same_line_and_material():
    rows = [history(index) for index in range(19)]
    rows += [history(index, line="光选线") for index in range(19)]
    rows += [history(index, material="PET") for index in range(19)]
    curves, fits = trained(rows)
    assert {fit.borrowed for fit in fits} == {3}
    assert set(curves) == {("涡电流线", "PP"), ("涡电流线", "PET")}


def test_lot_outside_post_overhaul_envelope_is_held_for_resampling():
    curves, _ = trained([history(index) for index in range(19)])
    curve = curves[("涡电流线", "PP")]
    inside = predict(lot(3400), audit(), curve)
    outside = predict(lot(500), audit(), curve)
    assert inside.rate is not None
    assert outside.rate is None and outside.notice == "料批超出检修后校准工况"


def test_non_overhauled_line_keeps_polymer_curve():
    curves, _ = trained([history(index) for index in range(19)])
    assert resolve(curves, "光选线", "PP") is None
    assert resolve(curves, "涡电流线", "PP", frozenset({"涡电流线"})) is not None


def test_late_closing_overhaul_sample_only_shifts_later_shifts():
    rows = [history(index) for index in range(19)]
    today = DAY + timedelta(days=20)
    record = (Overhaul("涡电流线", RESUME),)
    base, base_fits = calibrate(rows, DAY, today, record)
    late = RunHistory(
        RESUME,
        today + timedelta(days=3),
        "PP",
        model_vector(2500, 0.09, 0.05, "涡电流线"),
        0.80,
        "涡电流线",
    )
    same, same_fits = calibrate([*rows, late], DAY, today, record)
    assert base == same and base_fits[0].post_count == same_fits[0].post_count
    later, later_fits = calibrate([*rows, late], DAY, today + timedelta(days=5), record)
    assert later_fits[0].post_count == base_fits[0].post_count + 1


def test_clean_release_requires_calibrated_lower_bound_and_updates_bunker():
    from sortyield.grade_acceptance.engine import classify
    from sortyield.bunker_slots.contracts import Bunker, PlacementFlag
    from sortyield.bunker_slots.engine import allocate

    curves, _ = trained([history(index) for index in range(19)])
    overhauled = frozenset({"涡电流线"})
    bins = [Bunker("BN", "PP", 10000, 0)]

    def graded(kg):
        batch = lot(kg)
        sample = audit()
        estimate = predict(
            batch, sample, resolve(curves, batch.line, batch.material, overhauled)
        )
        return classify(batch, sample, estimate)

    released = graded(3400)
    assert released.grade == "清洁颗粒" and released.reserve_kg > 0
    book = allocate([released], bins, 10000)
    assert book.events[0].target == "BN" and book.events[0].flag == PlacementFlag.RESERVED

    gated = graded(500)
    assert gated.grade == "留样复检"
    assert allocate([gated], bins, 10000).events[0].target is None
