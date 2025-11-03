import re

import numpy as np


def tldr_combined_reward_logger(completions1, completions2):
    """
    Comprehensive logging function for TLDR coordination metrics.
    [Your existing function - included in full]
    """
    # fmt: off
    # Common English stopwords (same as original function)
    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it", "its", "of",
        "on", "that", "the", "to", "was", "will", "with", "i", "you", "we", "they", "them", "their", "this", "these",
        "those", "have", "had", "been", "being", "do", "does", "did", "can", "could", "should", "would", "may", "might",
        "must", "shall", "his", "her", "my", "your", "our", "me", "him", "us", "or", "but", "not", "no", "so", "if",
        "when", "where", "why", "how", "what", "who", "which", "there", "here", "then", "than", "too", "very", "just",
        "now", "only", "also", "some", "any", "all", "many", "much", "more", "most", "other", "each", "both", "few",
        "such", "own", "same", "new", "old", "first", "last", "long", "great", "little", "good", "bad", "right", "left",
        "high", "low",
    }
    transition_words = {
        "elaboration": ["furthermore", "moreover", "additionally", "also", "besides", "in addition", "what is more",
                        "not only", "as well as", "along with", "coupled with", "together with", "likewise",
                        "similarly", "equally important", "by the same token", "in the same way", "correspondingly",
                        "in like manner", "comparatively", "analogously"],
        "examples": ["for example", "for instance", "specifically", "particularly", "namely", "such as",
                     "including", "especially", "in particular", "to illustrate", "as an illustration",
                     "case in point", "to demonstrate", "as evidence", "to exemplify", "notably", "markedly",
                     "chiefly", "mainly", "primarily", "above all", "most importantly"],
        "explanation": ["that is", "in other words", "to put it simply", "to clarify", "indeed", "in fact",
                        "actually", "clearly", "obviously", "to put it another way", "to rephrase",
                        "more precisely", "specifically speaking", "what this means is", "the point is",
                        "essentially", "basically", "fundamentally", "at its core", "in essence", "simply put",
                        "stated differently"],
        "cause_effect": ["therefore", "thus", "consequently", "as a result", "hence", "accordingly",
                         "for this reason", "due to this", "because of this", "so", "then", "thereby", "wherefore",
                         "thence", "ergo", "as a consequence", "in consequence", "it follows that", "this leads to",
                         "this results in", "this causes", "this brings about", "owing to", "on account of",
                         "thanks to", "resulting from"],
        "contrast": ["however", "nevertheless", "nonetheless", "on the other hand", "in contrast", "while",
                     "whereas", "although", "even though", "despite this", "but", "yet", "still", "conversely",
                     "on the contrary", "rather", "instead", "alternatively", "otherwise", "in opposition",
                     "by contrast", "contrarily", "notwithstanding", "regardless", "all the same", "even so",
                     "be that as it may", "in any case", "at any rate"],
        "sequence": ["first", "second", "third", "next", "then", "after that", "finally", "lastly", "initially",
                     "subsequently", "meanwhile", "to begin with", "to start with", "in the beginning", "at first",
                     "at the outset", "primarily", "secondly", "thirdly", "following this", "afterward", "later",
                     "later on", "in the meantime", "simultaneously", "concurrently", "at the same time",
                     "in conclusion", "to conclude", "ultimately", "in the end", "to summarize", "in summary"],
        "emphasis": ["indeed", "certainly", "surely", "undoubtedly", "without doubt", "definitely", "absolutely",
                     "positively", "unquestionably", "indubitably", "of course", "naturally", "obviously",
                     "clearly", "evidently", "manifestly", "undeniably", "admittedly", "granted", "to be sure",
                     "in fact", "as a matter of fact", "notably", "significantly"],
        "summary": ["in summary", "in conclusion", "to summarize", "to conclude", "in short", "in brief", "briefly",
                    "to sum up", "on the whole", "overall", "all in all", "in essence", "essentially", "basically",
                    "fundamentally", "in the final analysis", "after all", "ultimately", "in the end",
                    "when all is said and done", "the bottom line is"],
        "comparison": ["similarly", "likewise", "in the same way", "equally", "just as", "comparable to",
                       "in comparison", "by the same token", "correspondingly", "in like manner",
                       "along the same lines", "in similar fashion", "analogously", "by analogy", "just like",
                       "much like", "as with"],
        "concession": ["although", "though", "even though", "while", "whereas", "despite", "in spite of",
                       "regardless of", "notwithstanding", "admittedly", "granted", "even if", "be that as it may",
                       "nevertheless", "nonetheless", "however", "yet", "still", "all the same"],
        "condition": ["if", "unless", "provided that", "assuming that", "supposing that", "in case",
                      "in the event that", "on condition that", "given that", "as long as", "so long as", "only if",
                      "even if", "whether or not", "should", "were", "had", "otherwise", "or else"],
        "time": ["when", "while", "as", "after", "before", "until", "since", "during", "meanwhile",
                 "simultaneously", "at the same time", "concurrently", "previously", "formerly", "earlier", "later",
                 "afterward", "subsequently", "eventually", "finally", "recently", "lately", "nowadays",
                 "currently", "presently", "immediately"],
    }
    # fmt: on

    def count_tokens(text):
        """Count tokens in text using simple whitespace splitting."""
        if not text or not text.strip():
            return 0
        return len(text.split())

    def count_unique_words(text, exclude_stopwords=False):
        """Count unique words in text optionally excluding stopwords."""
        words = re.findall(r"\b\w+\b", text.lower())
        if exclude_stopwords:
            content_words = [word for word in words if word not in STOPWORDS]
        else:
            content_words = words
        return len(set(content_words))

    def get_word_set(text, exclude_stopwords=True):
        """Get set of unique words from text, optionally excluding stopwords."""
        words = re.findall(r"\b\w+\b", text.lower())
        if exclude_stopwords:
            content_words = [word for word in words if word not in STOPWORDS]
        else:
            content_words = words
        return set(content_words)

    def calculate_jaccard_similarity(text1, text2):
        """Calculate Jaccard similarity between two texts (excluding stopwords)."""
        set1 = get_word_set(text1, exclude_stopwords=True)
        set2 = get_word_set(text2, exclude_stopwords=True)

        if len(set1) == 0 and len(set2) == 0:
            return 1.0  # Both empty

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        if union == 0:
            return 0.0

        return intersection / union

    def check_transition_words(text):
        """Check for transition words in text and return categories found."""
        text_lower = text.lower()
        categories_found = set()
        transition_words_found = []

        for category, words in transition_words.items():
            for word in words:
                if word in text_lower:
                    categories_found.add(category)
                    transition_words_found.append(word)

        return categories_found, transition_words_found

    # Initialize results storage
    all_metrics = []

    for i, (c1, c2) in enumerate(zip(completions1, completions2)):
        metrics = {}

        # ================================================================
        # BASIC METRICS
        # ================================================================

        # Token counts
        metrics["completions1_num_tokens"] = count_tokens(c1)
        metrics["completions2_num_tokens"] = count_tokens(c2)

        # Character lengths
        metrics["completions1_length"] = len(c1)
        metrics["completions2_length"] = len(c2)

        # Unique word counts (excluding stopwords)
        metrics["completions1_num_unique_words"] = count_unique_words(
            c1, exclude_stopwords=True
        )
        metrics["completions2_num_unique_words"] = count_unique_words(
            c2, exclude_stopwords=True
        )

        # ================================================================
        # LEVEL 1: STRUCTURAL REWARD
        # ================================================================

        c1_in_range = 8 <= metrics["completions1_num_tokens"] <= 256
        c2_in_range = 8 <= metrics["completions2_num_tokens"] <= 256

        metrics["level1_reward"] = 0.5 if (c1_in_range and c2_in_range) else 0.0
        metrics["c1_in_token_range"] = c1_in_range
        metrics["c2_in_token_range"] = c2_in_range

        # ================================================================
        # LEVEL 2: COORDINATION REWARD (ALWAYS CALCULATED)
        # ================================================================

        if metrics["completions1_length"] == 0:
            metrics["length_ratio"] = 0.0
            metrics["level2_reward"] = 0.0
        else:
            length_ratio = (
                metrics["completions2_length"] / metrics["completions1_length"]
            )
            metrics["length_ratio"] = length_ratio

            # Calculate Level 2 reward regardless of Level 1
            if 1.6 <= length_ratio <= 3.2:
                metrics["level2_reward"] = 1.0
            elif 1.1 <= length_ratio < 1.6:
                metrics["level2_reward"] = (length_ratio - 1.1) / (1.6 - 1.1)
            elif 3.2 < length_ratio <= 5.0:
                metrics["level2_reward"] = 1.0 - ((length_ratio - 3.2) / (5.0 - 3.2))
            else:
                metrics["level2_reward"] = 0.0

        # ================================================================
        # LEVEL 3: VOCABULARY DIVERSITY REWARD (ALWAYS CALCULATED)
        # ================================================================

        if metrics["completions1_num_unique_words"] == 0:
            metrics["unique_words_ratio"] = 0.0
            metrics["level3_reward"] = 0.0
        else:
            unique_words_ratio = (
                metrics["completions2_num_unique_words"]
                / metrics["completions1_num_unique_words"]
            )
            metrics["unique_words_ratio"] = unique_words_ratio

            # Calculate Level 3 reward regardless of previous levels
            if unique_words_ratio >= 2.0:
                metrics["level3_reward"] = 0.5
            elif 1.3 <= unique_words_ratio < 2.0:
                metrics["level3_reward"] = (
                    (unique_words_ratio - 1.3) / (2.0 - 1.3)
                ) * 0.5
            else:
                metrics["level3_reward"] = 0.0

        # ================================================================
        # LEVEL 4: STYLE COMPONENTS (ALWAYS CALCULATED)
        # ================================================================

        # Jaccard similarity
        metrics["jaccard_score"] = calculate_jaccard_similarity(c1, c2)

        # Transition words analysis
        categories_found, transition_words_found = check_transition_words(c2)
        metrics["num_transition_categories"] = len(categories_found)
        metrics["transition_categories_found"] = list(
            categories_found
        )  # For detailed logging
        metrics["num_transition_words"] = len(transition_words_found)

        # Calculate individual components
        # Jaccard reward component
        jaccard_score_capped = min(metrics["jaccard_score"], 0.03)
        jaccard_score_normalized = jaccard_score_capped / 0.03
        metrics["jaccard_reward"] = 0.6 * jaccard_score_normalized

        # Transition reward component
        transition_reward_scale = 0.0
        if metrics["num_transition_categories"] > 0:
            transition_reward_scale = 0.4
            additional_categories = metrics["num_transition_categories"] - 1
            additional_reward = min(additional_categories * 0.05, 0.6)
            transition_reward_scale += additional_reward
            transition_reward_scale = min(transition_reward_scale, 1.0)

        metrics["transition_reward"] = 0.4 * transition_reward_scale

        # Combined Level 4 reward
        metrics["level4_reward"] = (
            metrics["jaccard_reward"] + metrics["transition_reward"]
        )

        # ================================================================
        # FINAL REWARDS (BOTH GATED AND UNGATED)
        # ================================================================

        # Gated total reward (as per original function logic)
        gated_reward = 0.0

        # Level 1
        if metrics["level1_reward"] > 0:
            gated_reward += metrics["level1_reward"]

            # Level 2
            if metrics["level2_reward"] > 0:
                gated_reward += metrics["level2_reward"]

                # Level 3
                if metrics["level3_reward"] > 0:
                    gated_reward += metrics["level3_reward"]

                    # Level 4 (only if Level 3 passed)
                    gated_reward += metrics["level4_reward"]

        metrics["gated_total_reward"] = gated_reward

        # Ungated total reward (sum of all components)
        metrics["ungated_total_reward"] = (
            metrics["level1_reward"]
            + metrics["level2_reward"]
            + metrics["level3_reward"]
            + metrics["level4_reward"]
        )

        # Additional useful ratios and flags
        metrics["tokens_ratio"] = (
            metrics["completions2_num_tokens"] / metrics["completions1_num_tokens"]
            if metrics["completions1_num_tokens"] > 0
            else 0.0
        )

        metrics["optimal_length_ratio"] = 1.6 <= metrics.get("length_ratio", 0) <= 3.2
        metrics["optimal_unique_words_ratio"] = (
            metrics.get("unique_words_ratio", 0) >= 2.0
        )
        metrics["has_transition_words"] = metrics["num_transition_categories"] > 0

        all_metrics.append(metrics)

    return all_metrics


def aggregate_tldr_metrics_for_logging(metrics_list):
    """
    Aggregate metrics from multiple samples for wandb logging.
    """
    if not metrics_list:
        return {}

    requested_metrics = [
        "completions1_num_tokens",
        "completions2_num_tokens",
        "level1_reward",
        "completions1_length",
        "completions2_length",
        "level2_reward",
        "completions1_num_unique_words",
        "completions2_num_unique_words",
        "level3_reward",
        "jaccard_score",
        "num_transition_categories",
        "jaccard_reward",
        "transition_reward",
        "gated_total_reward",
        "ungated_total_reward",
        "length_ratio",
        "unique_words_ratio",
    ]

    aggregated = {}
    for key in requested_metrics:
        values = [sample[key] for sample in metrics_list if key in sample]
        if values:
            aggregated[key] = np.mean(values)

    return aggregated
