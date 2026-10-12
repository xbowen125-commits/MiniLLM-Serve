"""Bounded serving-shaped workloads using deterministic virtual durations."""

from dataclasses import asdict, dataclass
import json

from minillm.contracts import ExecutionPhase
from .scheduler_compare import SchedulerComparisonRunner, SimulatedRequest


@dataclass(frozen=True, slots=True)
class StressScenario:
    name: str
    requests: tuple[SimulatedRequest, ...]
    start_time: float = 0.0
    starvation_threshold: float = 10.0
    aging_factor: float = 1.0

    def __post_init__(self):
        if not self.name or not 1 <= len(self.requests) <= 1000:
            raise ValueError("stress scenario requires 1..1000 requests and a name")
        SchedulerComparisonRunner(start_time=self.start_time,
                                  starvation_threshold=self.starvation_threshold,
                                  aging_factor=self.aging_factor)

    @classmethod
    def decode_flood(cls):
        return cls._from_config("decode_flood")

    @classmethod
    def mixed_length(cls):
        return cls._from_config("mixed_length")

    @classmethod
    def continuous_arrival(cls):
        return cls._from_config("continuous_arrival")

    @classmethod
    def _from_config(cls, name):
        from .report_generator import ROOT, build_scenario, load_config
        scenario = build_scenario(load_config(ROOT / "configs" / f"{name}.json"))
        # Retain the older factory display names for existing CLI consumers.
        return cls(scenario.name.replace("_", "-"), scenario.requests,
                   scenario.start_time, scenario.starvation_threshold, scenario.aging_factor)

    def run(self):
        runner = SchedulerComparisonRunner(start_time=self.start_time,
                                           starvation_threshold=self.starvation_threshold,
                                           aging_factor=self.aging_factor)
        return runner.run(self.requests)


def main():
    scenarios = (StressScenario.decode_flood(), StressScenario.mixed_length(),
                 StressScenario.continuous_arrival())
    print(json.dumps({
        "scope": "virtual-clock-admission-stress-simulation",
        "delay_unit": "seconds",
        "scenarios": {s.name: {"request_count": len(s.requests),
                               "start_time": s.start_time,
                               "starvation_threshold": s.starvation_threshold,
                               "aging_factor": s.aging_factor,
                               "results": {p: asdict(r) for p, r in s.run().items()}}
                      for s in scenarios},
    }, indent=2))


if __name__ == "__main__":
    main()
