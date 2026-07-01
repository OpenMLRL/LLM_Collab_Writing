from collections import defaultdict
from typing import Any, Dict, List

from loggers.arxiv_logger import (
    aggregate_arxiv_metrics_for_logging,
    arxiv_combined_reward_logger,
)
from loggers.tldr_logger import (
    aggregate_tldr_metrics_for_logging,
    tldr_combined_reward_logger,
)


def build_ac_writing_metrics_callback(dataset_type: str, num_agents: int):
    dataset_key = (dataset_type or "").lower()
    if dataset_key == "arxiv":
        logger = arxiv_combined_reward_logger
        aggregator = aggregate_arxiv_metrics_for_logging
    elif dataset_key == "tldr":
        logger = tldr_combined_reward_logger
        aggregator = aggregate_tldr_metrics_for_logging
    else:
        return None

    def callback(rollouts: List[Any]) -> Dict[str, float]:
        return aggregate_ac_writing_metrics(
            rollouts,
            num_agents=max(1, int(num_agents)),
            logger=logger,
            aggregator=aggregator,
        )

    return callback


def aggregate_ac_writing_metrics(
    rollouts: List[Any], *, num_agents: int, logger, aggregator
) -> Dict[str, float]:
    if not rollouts or num_agents < 2:
        return {}

    by_generation: Dict[int, Dict[int, str]] = defaultdict(dict)
    max_generation = 0
    for sample in rollouts:
        metadata = getattr(sample, "metadata", {}) or {}
        generation_idx = int(metadata.get("generation_idx", 0))
        agent_idx = int(getattr(sample, "agent_idx", 0))
        max_generation = max(max_generation, generation_idx)
        by_generation[generation_idx][agent_idx] = str(
            getattr(sample, "completion", "") or ""
        )

    completions1: List[str] = []
    completions2: List[str] = []
    for generation_idx in range(max_generation + 1):
        completions1.append(by_generation[generation_idx].get(0, ""))
        completions2.append(by_generation[generation_idx].get(1, ""))

    detailed = logger(completions1, completions2)
    return aggregator(detailed)
