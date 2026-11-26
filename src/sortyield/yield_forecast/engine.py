from collections import defaultdict
from dataclasses import replace
from math import ceil, fsum, log1p, sqrt
from .contracts import Curve, Estimate, OverhaulFit


def model_vector(kg, impurity, moisture, line):
    return log1p(kg), impurity, moisture, float(line == "涡电流线")


def normalize(raw, center, scale):
    return tuple(
        (x - mu) / sigma for x, mu, sigma in zip(raw, center, scale, strict=True)
    )


def center_fit(rows):
    columns = tuple(zip(*(row.features for row in rows), strict=True))
    center = tuple(fsum(col) / len(col) for col in columns)
    scale = tuple(
        max(1e-6, sqrt(fsum((x - mu) ** 2 for x in col) / len(col)))
        for col, mu in zip(columns, center, strict=True)
    )
    return center, scale, tuple((min(col), max(col)) for col in columns)


def clipped_prediction(vector, center, scale, params, base):
    weighted = fsum(
        w * x for w, x in zip(params, normalize(vector, center, scale), strict=True)
    )
    return max(0, min(1, base + weighted))


def fit_curve(material, train, validation, penalty=0.10):
    if len(train) < 5:
        raise ValueError("回归样本不足")
    center, scale, limits = center_fit(train)
    values = [normalize(row.features, center, scale) for row in train]
    target = [row.yield_rate for row in train]
    base = fsum(target) / len(target)
    params = [0.0] * len(center)
    for _ in range(150):
        errors = [
            base + fsum(w * x for w, x in zip(params, features, strict=True)) - y
            for features, y in zip(values, target, strict=True)
        ]
        gradient = [
            fsum(
                error * features[i]
                for error, features in zip(errors, values, strict=True)
            )
            / len(train)
            + penalty * params[i]
            for i in range(len(params))
        ]
        params = [
            weight - 0.1 * slope for weight, slope in zip(params, gradient, strict=True)
        ]
    residual = sorted(
        abs(
            row.yield_rate
            - clipped_prediction(row.features, center, scale, params, base)
        )
        for row in validation
    )
    quantile = (
        residual[min(len(residual), ceil(0.9 * (len(residual) + 1))) - 1]
        if residual
        else 0.12
    )
    return Curve(
        material,
        center,
        scale,
        limits,
        tuple(params),
        base,
        quantile,
        len(train),
        len(validation),
    )


def _baseline(history, fold, cutoff):
    train = defaultdict(list)
    calibration = defaultdict(list)
    for row in history:
        if row.handled < fold and row.closed <= fold:
            train[row.polymer].append(row)
        elif row.handled >= fold and row.closed <= cutoff:
            calibration[row.polymer].append(row)
    return {
        material: fit_curve(material, rows, calibration[material])
        for material, rows in sorted(train.items())
        if len(rows) >= 5
    }


def calibrate(history, fold, cutoff, overhauls=()):
    curves = _baseline(history, fold, cutoff)
    fits = []
    for record in overhauls:
        matured = [
            row for row in history if row.line == record.line and row.closed <= cutoff
        ]
        for material in sorted({row.polymer for row in matured}):
            fit = overhaul_fit(
                material, record, [row for row in matured if row.polymer == material]
            )
            fits.append(fit)
            if fit.curve is not None:
                curves[(record.line, material)] = fit.curve
    return curves, tuple(fits)


def per_polymer(history, fold, cutoff, overhauls=()):
    return calibrate(history, fold, cutoff, overhauls)[0]


def overhaul_fit(material, record, matured):
    ordered = sorted(matured, key=lambda row: (row.handled, row.closed))
    post = [row for row in ordered if row.handled >= record.resumed]
    prior = [row for row in ordered if row.handled < record.resumed]
    if len(post) < record.minimum:
        return OverhaulFit(
            record.line, material, None, len(post), 0, 0, "检修后未成熟样本不足"
        )
    calibration = post[-record.minimum :]
    recent = post[: -record.minimum]
    train = list(recent)
    borrowed = 0
    if len(train) < record.minimum:
        take = prior[-(record.minimum - len(train)) :]
        train = [*take, *train]
        borrowed = len(take)
    if len(train) < record.minimum:
        return OverhaulFit(
            record.line,
            material,
            None,
            len(post),
            len(calibration),
            borrowed,
            "检修后及同线同材质旧记录不足",
        )
    curve = fit_curve(material, train, calibration)
    envelope = tuple(
        (min(column), max(column))
        for column in zip(*(row.features for row in calibration), strict=True)
    )
    return OverhaulFit(
        record.line,
        material,
        replace(curve, envelope=envelope),
        len(post),
        len(calibration),
        borrowed,
        "检修后校准",
    )


def resolve(curves, line, material, overhauled=frozenset()):
    if line in overhauled:
        return curves.get((line, material))
    return curves.get((line, material), curves.get(material))


def predict(lot, audit, curve):
    if curve is None or not audit.approved_samples:
        return Estimate(lot.lot, None, None, None, lot.material, "缺足量抽检或拟合记录")
    vector = model_vector(lot.kg, audit.impurity, audit.moisture, lot.line)
    if curve.envelope is not None and any(
        x < low or x > high
        for x, (low, high) in zip(vector, curve.envelope, strict=True)
    ):
        return Estimate(
            lot.lot, None, None, None, curve.material, "料批超出检修后校准工况"
        )
    distance = max(
        max(low - x, x - high, 0) / sd
        for x, (low, high), sd in zip(vector, curve.limits, curve.scale, strict=True)
    )
    if distance > 2.5:
        return Estimate(lot.lot, None, None, None, curve.material, "料批超出训练支撑")
    score = clipped_prediction(
        vector, curve.center, curve.scale, curve.params, curve.rate
    )
    return Estimate(
        lot.lot,
        score,
        max(0, score - curve.loss_radius),
        min(1, score + curve.loss_radius),
        curve.material,
        "已估算",
    )
