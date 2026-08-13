"""Centralized comparator adapters for collaborative writing domains."""

from typing import Any, Dict, Sequence

from comlrl.trainers.preference import (
    CentralizedComparatorParseError,
    TaggedCentralizedComparatorAdapter,
)


class WritingCentralizedComparatorAdapter:
    def __init__(self, dataset_type: str):
        dataset_key = str(dataset_type).strip().lower()
        if dataset_key not in {"tldr", "arxiv"}:
            raise ValueError(f"Unsupported writing dataset type: {dataset_type}")
        self.dataset_type = dataset_key
        self._tagged_parser = TaggedCentralizedComparatorAdapter()

    def build_prompt(
        self,
        batch_item: Dict[str, Any],
        agent_prompts: Sequence[str],
    ) -> str:
        del batch_item
        if len(agent_prompts) != 2:
            raise ValueError(
                "Collaborative writing centralized generation requires 2 agents."
            )

        if self.dataset_type == "tldr":
            roles = (
                "Agent 0 must provide the concise summary response.",
                "Agent 1 must provide the detailed summary response.",
            )
            task_label = "two complementary summary writers"
        else:
            roles = (
                "Agent 0 must provide the background and motivation contribution.",
                "Agent 1 must provide the methodology and implications contribution.",
            )
            task_label = "two complementary scientific-writing agents"

        return f"""You are acting as one centralized coordinator for {task_label}.

Generate the exact two separate prose outputs that the decentralized agents would
submit. Follow each original prompt and keep the role assignments distinct. Do not
merge the responses, discuss the coordination process, or add text outside the tags.

{roles[0]}
Agent 0 original prompt:
{agent_prompts[0]}

{roles[1]}
Agent 1 original prompt:
{agent_prompts[1]}

Return exactly this structure:
<agent_0>
the complete Agent 0 prose response
</agent_0>
<agent_1>
the complete Agent 1 prose response
</agent_1>
"""

    def parse_completion(
        self,
        completion: str,
        batch_item: Dict[str, Any],
        num_agents: int,
    ) -> Sequence[str]:
        try:
            return self._tagged_parser.parse_completion(
                completion,
                batch_item,
                num_agents,
            )
        except CentralizedComparatorParseError:
            return [""] * num_agents


def get_writing_centralized_comparator_adapter(
    dataset_type: str,
) -> WritingCentralizedComparatorAdapter:
    return WritingCentralizedComparatorAdapter(dataset_type)
