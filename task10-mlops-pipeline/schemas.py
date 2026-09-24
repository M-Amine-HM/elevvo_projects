from typing import Dict, Literal

from pydantic import BaseModel, Field, model_validator


class SensorInput(BaseModel):
    """Raw AI4I sensor readings that feed the failure-risk model.

    Field names are the descriptive API names. ``product_type`` maps to Task 9's
    ``Type`` column; the numeric fields map to the air/process temperature,
    rotational speed, torque and tool-wear sensors. Task 9 derives four extra
    engineered features from these (temperature difference, thermal stress,
    torque-speed ratio, power proxy) — the API recomputes them server-side, never
    the client.
    """

    air_temperature: float = Field(
        ge=240.0,
        le=350.0,
        json_schema_extra={"example": 298.1},
        description=(
            "Air temperature in Kelvin. AI4I sensor range is roughly 295-304 K; "
            "values far outside that physical envelope are rejected as 400."
        ),
    )
    process_temperature: float = Field(
        ge=240.0,
        le=350.0,
        json_schema_extra={"example": 308.6},
        description="Process temperature in Kelvin (must sit above air temperature).",
    )
    rotational_speed: float = Field(
        gt=0.0,
        le=4000.0,
        json_schema_extra={"example": 1551},
        description=(
            "Rotational speed in rpm. Must be strictly positive — a zero or "
            "negative reading means the machine is not running."
        ),
    )
    torque: float = Field(
        ge=0.0,
        le=150.0,
        json_schema_extra={"example": 42.8},
        description="Torque in Nm.",
    )
    tool_wear: float = Field(
        ge=0.0,
        le=500.0,
        json_schema_extra={"example": 0.0},
        description="Tool wear in minutes of operation.",
    )
    product_type: Literal["L", "M", "H"] = Field(
        json_schema_extra={"example": "L"},
        description="Product quality variant (Task 9 'Type' column).",
    )

    @model_validator(mode="after")
    def _process_must_exceed_air(self) -> "SensorInput":
        if self.process_temperature <= self.air_temperature:
            raise ValueError(
                "process_temperature must be strictly above air_temperature "
                "(physically impossible otherwise)."
            )
        return self


class PredictionOutput(BaseModel):
    """Structured prediction returned by ``POST /predict``.

    ``predicted_failure_type`` follows Task 9's frozen decision rule: the argmax
    class is reported, unless it is a failure class whose probability falls below
    ``decision_threshold``, in which case the verdict becomes ``No failure``.
    """

    predicted_failure_type: str = Field(
        description=(
            "Predicted failure type after applying the decision threshold "
            "(one of Heat Dissipation Failure / No failure / Overstrain Failure / "
            "Power Failure / Random Failure / Tool Wear Failure)."
        ),
        json_schema_extra={"example": "No failure"},
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the reported class (its predicted probability).",
        json_schema_extra={"example": 0.9976},
    )
    risk: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Failure risk — 1 - P(No failure) when the verdict is No failure, "
            "otherwise the reported failure class probability."
        ),
        json_schema_extra={"example": 0.0024},
    )
    decision_threshold: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Frozen, validation-tuned action threshold used by Task 9's decision "
            "rule (reported for context, never recomputed here)."
        ),
        json_schema_extra={"example": 0.4496},
    )
    failure_probabilities: Dict[str, float] = Field(
        description=(
            "Per-class probability breakdown across all 6 classes, keyed by class "
            "name in model order: Heat Dissipation Failure, No failure, Overstrain "
            "Failure, Power Failure, Random Failure, Tool Wear Failure."
        ),
        json_schema_extra={
            "example": {
                "Heat Dissipation Failure": 0.0006,
                "No failure": 0.9976,
                "Overstrain Failure": 0.0005,
                "Power Failure": 0.0009,
                "Random Failure": 0.0004,
                "Tool Wear Failure": 0.0009,
            }
        },
    )