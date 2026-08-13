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

    def build_sequential_prompt(
        self,
        batch_item: Dict[str, Any],
        agent_prompts: Sequence[str],
        agent_index: int,
        previous_outputs: Sequence[str],
    ) -> str:
        del batch_item
        if len(agent_prompts) != 2:
            raise ValueError(
                "Collaborative writing centralized generation requires 2 agents."
            )
        if agent_index not in {0, 1} or len(previous_outputs) != agent_index:
            raise ValueError(
                "Sequential writing generation requires ordered Agent 0 then Agent 1."
            )

        if self.dataset_type == "tldr":
            roles = (
                "a concise summary response",
                "a detailed summary response that complements the concise response",
            )
            task_label = "two complementary summary writers"
        else:
            roles = (
                "the background and motivation contribution",
                "the methodology and implications contribution that complements the "
                "background",
            )
            task_label = "two complementary scientific-writing agents"

        earlier_context = (
            "No earlier contribution has been finalized. Agent 1 will receive your "
            "response before writing its contribution."
            if not previous_outputs
            else (
                "The following contribution is final. Do not rewrite it; make your "
                f"response complementary:\n<agent_0>\n{previous_outputs[0]}\n</agent_0>"
            )
        )
        return f"""You are Agent {agent_index} in a centralized sequential team of {task_label}.

The team is producing one joint response. You can inspect both role assignments and
all earlier finalized contributions. Write only {roles[agent_index]}.

Agent 0 original prompt:
{agent_prompts[0]}

Agent 1 original prompt:
{agent_prompts[1]}

Centralized context:
{earlier_context}

Return exactly this structure, with one complete prose response and no text outside it:
<agent_{agent_index}>
Agent {agent_index} prose response
</agent_{agent_index}>
"""

    def parse_sequential_completion(
        self,
        completion: str,
        batch_item: Dict[str, Any],
        agent_index: int,
    ) -> str:
        try:
            return self._tagged_parser.parse_sequential_completion(
                completion,
                batch_item,
                agent_index,
            )
        except CentralizedComparatorParseError:
            return str(completion).strip()


def get_writing_centralized_comparator_adapter(
    dataset_type: str,
) -> WritingCentralizedComparatorAdapter:
    return WritingCentralizedComparatorAdapter(dataset_type)
