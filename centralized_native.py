"""Native centralized writing prompt: one writer produces both responses.

The concatenated adapter embeds both agents' prompts, so the post or abstract
appears twice and the arXiv copies talk about "another agent". This prompt
states the source once and asks one writer for both tagged responses; the
unchanged tagged parser splits them back into the agents' outputs for the same
reward.
"""

from typing import Any, Dict, Sequence

from centralized_comparator import WritingCentralizedComparatorAdapter


def build_native_centralized_prompt(example: Dict[str, Any], dataset_type: str) -> str:
    if dataset_type == "tldr":
        post = example.get("prompt", "")
        if not post:
            # The role formatters return this message for the same rows.
            return "Error: No prompt provided."
        return f"""You are one writer. Write two complementary summary responses to this post
yourself: a concise summary and a detailed summary. Keep them as two separate
responses; do not merge them, discuss the process, or add text outside the tags.

Post:
{post}

Response agent_0 (concise summary):
- Provide a brief, focused summary in one sentence or a few sentences
- Be factual and informative

Response agent_1 (detailed summary):
- Use more unique words
- Use some transition words to improve flow

Return exactly this structure:
<agent_0>
the concise summary
</agent_0>
<agent_1>
the detailed summary
</agent_1>
"""
    abstract = example.get("abstract_text", "")
    if not abstract:
        return "Error: No abstract provided."
    return f"""You are one scientific writer. Based on the following scientific abstract, expand
content for an introduction section as two complementary parts that you write
yourself. Keep them as two separate responses; do not merge them, discuss the
process, or add text outside the tags.

Abstract:
{abstract}

Response agent_0 (background and motivation):
- Focus on background and motivation
- Avoid repeating methodology and implications content

Response agent_1 (methodology and implications):
- Focus on methodology and implications
- Avoid repeating background and motivation content

Return exactly this structure:
<agent_0>
the background and motivation part
</agent_0>
<agent_1>
the methodology and implications part
</agent_1>
"""


class WritingNativeCentralizedAdapter(WritingCentralizedComparatorAdapter):
    """State the source once; parse the same agent-tagged responses."""

    def build_prompt(
        self,
        batch_item: Dict[str, Any],
        agent_prompts: Sequence[str],
    ) -> str:
        if len(agent_prompts) != 2:
            raise ValueError(
                "Collaborative writing centralized generation requires 2 agents."
            )
        return build_native_centralized_prompt(batch_item, self.dataset_type)
