"""Generate synthetic contexts through the public offline facade."""

from minillm.engine import GenerationConfig
from minillm.runtime import RuntimeFacade, RuntimeRequestContext

from .scenarios import Scenario


def generate_workload(runtime: RuntimeFacade, scenario: Scenario) -> list[RuntimeRequestContext]:
    """Submit all requests before stepping; IDs and prompt contents are fixed."""
    prompt = " ".join(["synthetic"] * scenario.prompt_words)
    return [runtime.submit(prompt, GenerationConfig(max_tokens=limit, temperature=0),
                           request_id=f"request-{index:04d}")
            for index, limit in enumerate(scenario.token_limits)]
