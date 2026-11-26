from csv import DictReader
from datetime import date
from .contracts import Overhaul, RunHistory
from .engine import calibrate, model_vector, predict, resolve


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as source:
        rows = DictReader(source)
        return tuple(
            RunHistory(
                date.fromisoformat(row["handled"]),
                date.fromisoformat(row["closed"]),
                row["material"],
                model_vector(
                    int(row["kg"]),
                    float(row["impurity"]),
                    float(row["moisture"]),
                    row["line"],
                ),
                float(row["yield_rate"]),
                row["line"],
            )
            for row in rows
        )


def overhauls(path, cutoff):
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8-sig") as source:
        records = tuple(
            Overhaul(row["line"], date.fromisoformat(row["resumed"]))
            for row in DictReader(source)
        )
    if len({record.line for record in records}) != len(records):
        raise ValueError("同一产线重复登记检修")
    return tuple(record for record in records if record.resumed <= cutoff)


def failed(fits):
    return tuple(
        {"产线": fit.line, "材质": fit.material, "结果": fit.notice}
        for fit in fits
        if fit.curve is None
    )


def evidence(curves, forecasts, fits=()):
    def label(key):
        return "/".join(key) if isinstance(key, tuple) else key

    records = []
    for fit in fits:
        records.append(
            {
                "产线": fit.line,
                "材质": fit.material,
                "检修后成熟样本": fit.post_count,
                "参与校准": fit.calibration,
                "借用检修前旧记录": fit.borrowed,
                "结果": fit.notice,
            }
        )
    groups = {key[1] if isinstance(key, tuple) else key for key in curves}
    return {
        "title": "产率拟合",
        "summary": f"拟合 {len(groups)} 个材质，{sum(row.rate is not None for row in forecasts)} 批有区间",
        "records": [
            {
                "料批": row.lot,
                "材质": row.fit_group,
                "产率": row.rate,
                "下界": row.lower,
                "上界": row.upper,
                "结果": row.notice,
            }
            for row in forecasts
        ],
        "拟合记录数": {
            label(key): {
                "训练": model.count,
                "校准": model.validation,
                "检修后校准": model.envelope is not None,
            }
            for key, model in curves.items()
        },
        "检修校准": records,
        "未成模": failed(fits),
    }
