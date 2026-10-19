from pydantic import BaseModel, Field, model_validator

POLICIES = {"ceap": "CEAP", "immediate": "Immediate swapping", "fixed": "Fixed purification", "threshold": "Fixed threshold", "weakest": "Weakest-link first", "ablation": "Local-quality ablation", "oracle": "Oracle diagnostic"}


class LinkConfig(BaseModel):
    fidelity: float = Field(default=0.94, ge=0.25, le=1)
    success_probability: float = Field(default=0.60, ge=0, le=1)
    attempt_ms: float = Field(default=10, gt=0, le=10000)
    classical_ms: float = Field(default=1, ge=0, le=1000)
    model: str = "werner"

    @model_validator(mode="after")
    def validate_model(self):
        if self.model not in ("werner", "phase_flip", "amplitude_damping"):
            raise ValueError("Unsupported link model")
        return self


class Config(BaseModel):
    nodes: int = Field(default=4, ge=2, le=5)
    links: list[LinkConfig] = Field(default_factory=lambda: [LinkConfig(fidelity=f) for f in (.94, .76, .96)])
    policy: str = "ceap"
    gate_noise: float = Field(default=.005, ge=0, le=1)
    readout_error: float = Field(default=.005, ge=0, le=1)
    memory_ms: float = Field(default=500, ge=0, le=1e9)
    memories: int = Field(default=4, ge=2, le=16)
    attempt_budget: int = Field(default=60, ge=1, le=1000)
    deadline_ms: float = Field(default=220, gt=0, le=100000)
    target_fidelity: float = Field(default=.90, ge=0, le=1)
    local_threshold: float = Field(default=.90, ge=.25, le=1)
    gate_ms: float = Field(default=.01, gt=0, le=10)
    measurement_ms: float = Field(default=.05, gt=0, le=10)
    calibration_sigma: float = Field(default=0, ge=0, le=.25)
    rollouts: int = Field(default=32, ge=1, le=128)
    time_weight: float = Field(default=.05, ge=0, le=1)
    resource_weight: float = Field(default=.05, ge=0, le=1)
    seed: int = Field(default=42, ge=0, le=2**31-1)
    input_state: str = "+"
    theta: float = Field(default=1.5707963267948966, ge=0, le=3.141592653589793)
    phi: float = Field(default=0, ge=0, le=6.283185307179586)

    @model_validator(mode="after")
    def check(self):
        if len(self.links) != self.nodes - 1:
            raise ValueError("The number of links must equal nodes − 1")
        if self.policy not in POLICIES:
            raise ValueError("Unknown policy")
        if self.input_state not in ("0", "1", "+", "-", "+i", "-i", "custom"):
            raise ValueError("Unknown input state")
        return self


def presets():
    base = Config()
    def make(name, title, text, config):
        return {"id": name, "name": title, "description": text, "config": config.model_dump()}
    ideal = base.model_copy(deep=True)
    ideal.gate_noise = ideal.readout_error = ideal.memory_ms = 0
    for link in ideal.links:
        link.fidelity = link.success_probability = 1
    homogeneous = base.model_copy(deep=True)
    for link in homogeneous.links:
        link.fidelity = .85
    short = base.model_copy(deep=True); short.memory_ms = 40
    constrained = base.model_copy(deep=True); constrained.attempt_budget = 12; constrained.deadline_ms = 80
    return [make("heterogeneous", "Uneven links", "One weak junction between two cleaner links.", base),
            make("ideal", "Ideal reference", "No gate, readout, link, or memory errors.", ideal),
            make("homogeneous", "Balanced chain", "Three equally noisy elementary links.", homogeneous),
            make("short_memory", "Short memory", "Stored qubits decay on a 40 ms time scale.", short),
            make("constrained", "Tight resources", "Twelve attempts and an 80 ms deadline.", constrained)]
