from dataclasses import dataclass
from datetime import date
from math import isfinite


@dataclass(frozen=True)
class RunHistory:
    handled: date
    closed: date
    polymer: str
    features: tuple[float, ...]
    yield_rate: float
    line: str = ""

    def __post_init__(self):
        if self.closed <= self.handled or not self.polymer or len(self.features) != 4:
            raise ValueError("产率样本时序、品种或维度错误")
        if not all(isfinite(x) for x in self.features) or not 0 <= self.yield_rate <= 1:
            raise ValueError("产率观测值越界")


@dataclass(frozen=True)
class Overhaul:
    line: str
    resumed: date
    minimum: int = 5

    def __post_init__(self):
        if not self.line or self.minimum < 5:
            raise ValueError("检修产线、复工日或校准样本下限错误")


@dataclass(frozen=True)
class Curve:
    material: str
    center: tuple[float, ...]
    scale: tuple[float, ...]
    limits: tuple[tuple[float, float], ...]
    params: tuple[float, ...]
    rate: float
    loss_radius: float
    count: int
    validation: int
    envelope: tuple[tuple[float, float], ...] | None = None


@dataclass(frozen=True)
class Estimate:
    lot: str
    rate: float | None
    lower: float | None
    upper: float | None
    fit_group: str
    notice: str


@dataclass(frozen=True)
class OverhaulFit:
    line: str
    material: str
    curve: Curve | None
    post_count: int
    calibration: int
    borrowed: int
    notice: str
