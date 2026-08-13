import pytest

from centralized_comparator import WritingCentralizedComparatorAdapter


@pytest.mark.parametrize("dataset_type", ["tldr", "arxiv"])
def test_writing_prompt_and_parser_are_prose_specific(dataset_type):
    adapter = WritingCentralizedComparatorAdapter(dataset_type)
    prompt = adapter.build_prompt({}, ["first writing prompt", "second prompt"])
    assert "prose outputs" in prompt
    assert "<agent_0>" in prompt
    assert "def aux" not in prompt

    outputs = adapter.parse_completion(
        "<agent_0>short response</agent_0>" "<agent_1>detailed response</agent_1>",
        {},
        2,
    )
    assert outputs == ["short response", "detailed response"]


@pytest.mark.parametrize("dataset_type", ["tldr", "arxiv"])
def test_writing_sequential_prompt_conditions_second_agent(dataset_type):
    adapter = WritingCentralizedComparatorAdapter(dataset_type)
    first_prompt = adapter.build_sequential_prompt(
        {},
        ["first writing prompt", "second writing prompt"],
        0,
        [],
    )
    second_prompt = adapter.build_sequential_prompt(
        {},
        ["first writing prompt", "second writing prompt"],
        1,
        ["final first contribution"],
    )

    assert "<agent_0>" in first_prompt
    assert "final first contribution" in second_prompt
    assert "<agent_1>" in second_prompt
    assert (
        adapter.parse_sequential_completion(
            "<agent_1>complementary response</agent_1>",
            {},
            1,
        )
        == "complementary response"
    )
