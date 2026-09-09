"""One author produces both paragraphs; no role dispatcher or coordinator."""
import re
from rewards import tldr_rewards

DOMAIN = "tldr"
METRICS = ("reward", "parse_success", "length_pass", "length_ratio_pass", "vocabulary_ratio_pass")
dataset_defaults = dict(dataset="trl-lib/tldr", revision="21233da376667088e6eb1ce4ce19ed832c2935d3",
    train_samples=250, eval_samples=250, splits={"train": "train[:250]", "eval": "test[:250]"},
    online_epochs=8, dpo_epochs=1, eval_every_responses=320, periodic_eval_samples=16, project="TLDR",
    reward_note="Existing two-paragraph TLDR structural/style reward divided by 3. Not a semantic factuality or human-quality metric.")


def normalize_rows(rows, split):
    # Drop completion / annotated summary; never fed to the actor or evaluator.
    return [{"id": "tldr-" + split + "-" + str(i), "prompt": r["prompt"]} for i, r in enumerate(rows)]


def messages(row):
    return [{"role": "system", "content": (
        "Write the complete answer yourself: exactly two complementary paragraphs summarizing the Reddit post. "
        "First give a concise summary; then expand it with relevant details, more varied vocabulary and natural "
        "transitions, maintaining the same meaning and style. Make the second paragraph 2-3 times as long "
        "in characters as the first. Each paragraph must have 10-200 words. Do not invent facts. "
        "Separate the two paragraphs with one blank line. No labels, reasoning or delegation.")},
        {"role": "user", "content": row["prompt"]}]


def parse(text):
    marker_count = text.count("[PARAGRAPH_SPLIT]")
    if marker_count > 1:
        return None
    # Natural paragraph boundaries are already a complete direct answer. Retain
    # the legacy explicit delimiter as an equally unambiguous alternative.
    parts = text.split("[PARAGRAPH_SPLIT]") if marker_count == 1 else re.split(r"\n\s*\n", text.strip())
    if len(parts) != 2:
        return None
    paragraphs = [re.sub(r"^Paragraph\s+[12]:\s*", "", s.strip(), flags=re.I)
                  for s in parts]
    return tuple(paragraphs) if all(paragraphs) else None


def score(text, row):
    result = dict.fromkeys(METRICS, 0.0)
    parts = parse(text)
    if parts is None:
        return result
    a, b = parts
    result["parse_success"] = 1.0
    result["length_pass"] = float(all(8 <= len(p.split()) <= 256 for p in parts))
    result["length_ratio_pass"] = float(1.6 <= len(b) / len(a) <= 3.2)
    words = [set(re.findall(r"\b\w+\b", p.lower())) for p in parts]
    result["vocabulary_ratio_pass"] = float(len(words[0]) > 0 and len(words[1]) / len(words[0]) >= 2)
    # Only this independent process uses the original pure reward function.
    tldr_rewards.VERBOSE = False
    result["reward"] = tldr_rewards.tldr_combined_reward([a], [b])[0] / 3.0
    return result
