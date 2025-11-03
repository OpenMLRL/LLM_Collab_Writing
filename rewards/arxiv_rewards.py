def arxiv_combined_reward(completions1, completions2):
    """Level-based reward function for arXiv abstract expansion coordination.

    Level 1 (Structural): Check individual token counts (128-256 tokens each)
    - If both completions in range: reward = 0.5, proceed to Level 2
    - If either completion out of range: reward = 0, stop here

    Level 2 (Coordination): Check length ratio (completion2/completion1) - LENIENT
    - If 1.0-1.3x: reward = 1.0, proceed to Level 3 (LENIENT OPTIMAL)
    - If 0.8-1.5x: reward = 0-1 proportionally, proceed to Level 3 (LENIENT ACCEPTABLE)
    - If outside 0.8-1.5x: reward = 0, stop here

    Level 3 (Vocabulary Diversity): Check unique words ratio (excluding stopwords) - STRICTER
    - If 0.7-1.3x: reward = 0.5, proceed to Level 4
    - If 0.5-1.7x: reward = 0-0.5 proportionally, proceed to Level 4
    - If outside 0.5-1.7x: reward = 0, stop here

    Level 4 (Style Reward): Combined transition words and Jaccard similarity
    - Transition component (0.4 weight): Base 0.4 for 1+ categories, +0.05 per additional category (max 1.0)
    - Jaccard similarity component (0.6 weight): Based on vocabulary overlap (excluding stopwords)
    - Only accessible if Level 3 reward > 0

    Args:
        completions1: List of text completions from agent 1 (background/motivation)
        completions2: List of text completions from agent 2 (complementary content)

    Returns:
        List of final rewards

    Maximum reward: 3.0 (0.5 structural + 1.0 coordination + 0.5 vocabulary + 1.0 style)
    """
    import re

    # fmt: off
    STOPWORDS = {
        "a", "all", "also", "an", "and", "any", "are", "as", "at", "bad", "be", "been", "being",
        "both", "but", "by", "can", "could", "did", "do", "does", "each", "few", "first", "for",
        "from", "good", "great", "had", "has", "have", "he", "her", "here", "him", "his", "how",
        "i", "if", "in", "is", "it", "its", "just", "last", "left", "little", "long", "low", "many",
        "may", "me", "might", "more", "most", "much", "must", "my", "new", "no", "not", "now", "of",
        "old", "on", "only", "or", "other", "our", "own", "right", "same", "shall", "should", "so",
        "some", "such", "than", "that", "the", "their", "them", "then", "there", "these", "they",
        "this", "those", "to", "too", "us", "very", "was", "we", "were", "what", "when", "where",
        "which", "who", "why", "will", "with", "would", "you", "your",
    }
    transition_words = {
        "elaboration": [
            "furthermore", "moreover", "additionally", "also", "besides", "in addition",
            "what is more", "not only", "as well as", "along with", "coupled with",
            "together with", "likewise", "similarly", "equally important", "by the same token",
            "in the same way", "correspondingly", "in like manner", "comparatively", "analogously",
        ],
        "examples": [
            "for example", "for instance", "specifically", "particularly", "namely", "such as",
            "including", "especially", "in particular", "to illustrate", "as an illustration",
            "case in point", "to demonstrate", "as evidence", "to exemplify", "notably",
            "markedly", "chiefly", "mainly", "primarily", "above all", "most importantly",
        ],
        "explanation": [
            "that is", "in other words", "to put it simply", "to clarify", "indeed", "in fact",
            "actually", "clearly", "obviously", "to put it another way", "to rephrase",
            "more precisely", "specifically speaking", "what this means is", "the point is",
            "essentially", "basically", "fundamentally", "at its core", "in essence", "simply put",
            "stated differently",
        ],
        "cause_effect": [
            "therefore", "thus", "consequently", "as a result", "hence", "accordingly",
            "for this reason", "due to this", "because of this", "so", "then", "thereby",
            "wherefore", "thence", "ergo", "as a consequence", "in consequence", "it follows that",
            "this leads to", "this results in", "this causes", "this brings about", "owing to",
            "on account of", "thanks to", "resulting from",
        ],
        "contrast": [
            "however", "nevertheless", "nonetheless", "on the other hand", "in contrast", "while",
            "whereas", "although", "even though", "despite this", "but", "yet", "still",
            "conversely", "on the contrary", "rather", "instead", "alternatively", "otherwise",
            "in opposition", "by contrast", "contrarily", "notwithstanding", "regardless",
            "all the same", "even so", "be that as it may", "in any case", "at any rate",
        ],
        "sequence": [
            "first", "second", "third", "next", "then", "after that", "finally", "lastly",
            "initially", "subsequently", "meanwhile", "to begin with", "to start with",
            "in the beginning", "at first", "at the outset", "primarily", "secondly", "thirdly",
            "following this", "afterward", "later", "later on", "in the meantime", "simultaneously",
            "concurrently", "at the same time", "in conclusion", "to conclude", "ultimately",
            "in the end", "to summarize", "in summary",
        ],
        "emphasis": [
            "indeed", "certainly", "surely", "undoubtedly", "without doubt", "definitely",
            "absolutely", "positively", "unquestionably", "indubitably", "of course", "naturally",
            "obviously", "clearly", "evidently", "manifestly", "undeniably", "admittedly",
            "granted", "to be sure", "in fact", "as a matter of fact", "notably", "significantly",
        ],
        "summary": [
            "in summary", "in conclusion", "to summarize", "to conclude", "in short", "in brief",
            "briefly", "to sum up", "on the whole", "overall", "all in all", "in essence",
            "essentially", "basically", "fundamentally", "in the final analysis", "after all",
            "ultimately", "in the end", "when all is said and done", "the bottom line is",
        ],
        "comparison": [
            "similarly", "likewise", "in the same way", "equally", "just as", "comparable to",
            "in comparison", "by the same token", "correspondingly", "in like manner",
            "along the same lines", "in similar fashion", "analogously", "by analogy",
            "just like", "much like", "as with",
        ],
        "concession": [
            "although", "though", "even though", "while", "whereas", "despite", "in spite of",
            "regardless of", "notwithstanding", "admittedly", "granted", "even if",
            "be that as it may", "nevertheless", "nonetheless", "however", "yet", "still",
            "all the same",
        ],
        "condition": [
            "if", "unless", "provided that", "assuming that", "supposing that", "in case",
            "in the event that", "on condition that", "given that", "as long as", "so long as",
            "only if", "even if", "whether or not", "should", "were", "had", "otherwise",
            "or else",
        ],
        "time": [
            "when", "while", "as", "after", "before", "until", "since", "during", "meanwhile",
            "simultaneously", "at the same time", "concurrently", "previously", "formerly",
            "earlier", "later", "afterward", "subsequently", "eventually", "finally", "recently",
            "lately", "nowadays", "currently", "presently", "immediately",
        ],
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

    rewards = []

    for i, (c1, c2) in enumerate(zip(completions1, completions2)):
        reward = 0.0

        print("\n" + "=" * 60)
        print("🏆 TESTING ARXIV EXPANSION REWARD")
        print("=" * 60)
        print(f"📝 Sample {i}")
        print(f"🎯 Maximum possible reward: 3.0")

        # Calculate individual and combined token counts
        token_count1 = count_tokens(c1)
        token_count2 = count_tokens(c2)
        total_token_count = token_count1 + token_count2

        # Calculate unique words for both completions
        unique_words1 = count_unique_words(c1)
        unique_words2 = count_unique_words(c2)
        unique_words_ratio = unique_words2 / unique_words1 if unique_words1 > 0 else 0.0

        # ================================================================
        # LEVEL 1: STRUCTURAL REQUIREMENTS (INDIVIDUAL TOKEN COUNTS)
        # ================================================================
        print("\n📋 LEVEL 1: STRUCTURAL REQUIREMENTS")
        print("-" * 50)
        print(f"📊 Completion 1 token count: {token_count1}")
        print(f"📊 Completion 2 token count: {token_count2}")
        print(f"🎯 Required range for BOTH: 128-256 tokens")

        # Check if both completions are within range
        c1_in_range = 128 <= token_count1 <= 256
        c2_in_range = 128 <= token_count2 <= 256

        print(f"✅ Completion 1 in range: {c1_in_range}")
        print(f"✅ Completion 2 in range: {c2_in_range}")

        if c1_in_range and c2_in_range:
            level1_reward = 0.5
            reward += level1_reward
            print(f"✅ Both completions in range: +{level1_reward} (total: {reward})")
            print("➡️  Proceeding to Level 2")
        else:
            if not c1_in_range:
                print(f"❌ Completion 1 out of range: {token_count1} (need 128-256)")
            if not c2_in_range:
                print(f"❌ Completion 2 out of range: {token_count2} (need 128-256)")
            print("⏹️  STOPPING: Structural requirements not met")
            print(f"🏆 Final reward: {reward}")
            rewards.append(reward)
            continue

        # ================================================================
        # LEVEL 2: COORDINATION REQUIREMENTS (LENGTH RATIO) - LENIENT
        # ================================================================
        print("\n⚙️  LEVEL 2: COORDINATION REQUIREMENTS (LENIENT)")
        print("-" * 50)

        len1, len2 = len(c1), len(c2)
        print(f"📏 Completion 1 length: {len1} chars")
        print(f"📏 Completion 2 length: {len2} chars")

        if len1 == 0:
            print("❌ Empty completion1 - cannot calculate ratio")
            print("⏹️  STOPPING: Cannot evaluate coordination")
            print(f"🏆 Final reward: {reward}")
            rewards.append(reward)
            continue

        length_ratio = len2 / len1
        print(f"📊 Length ratio (C2/C1): {length_ratio:.2f}")
        print(f"🎯 Optimal range: 1.0-1.3x (LENIENT)")
        print(f"📈 Acceptable range: 0.8-1.5x (LENIENT)")

        level2_reward = 0.0
        proceed_to_level3 = False

        if 1.0 <= length_ratio <= 1.3:
            # Perfect range - maximum Level 2 reward (LENIENT)
            level2_reward = 1.0
            reward += level2_reward
            proceed_to_level3 = True
            print(f"✅ Perfect length ratio: +{level2_reward} (total: {reward})")
            print("🎉 OPTIMAL COORDINATION ACHIEVED!")
            print("➡️  Proceeding to Level 3")
        elif 0.8 <= length_ratio < 1.0:
            # Linear interpolation from 0 to 1 between 0.8 and 1.0 (LENIENT)
            level2_reward = (length_ratio - 0.8) / (1.0 - 0.8)
            reward += level2_reward
            proceed_to_level3 = True
            print(f"⚠️  Below optimal ratio: +{level2_reward:.2f} (total: {reward})")
            print("💡 Consider making completion2 longer for better coordination")
            print("➡️  Proceeding to Level 3")
        elif 1.3 < length_ratio <= 1.5:
            # Linear interpolation from 1 to 0 between 1.3 and 1.5 (LENIENT)
            level2_reward = 1.0 - ((length_ratio - 1.3) / (1.5 - 1.3))
            reward += level2_reward
            proceed_to_level3 = True
            print(f"⚠️  Above optimal ratio: +{level2_reward:.2f} (total: {reward})")
            print("💡 Consider making completion2 shorter for better coordination")
            print("➡️  Proceeding to Level 3")
        else:
            # Outside acceptable range - no Level 2 reward
            level2_reward = 0.0
            proceed_to_level3 = False
            print(f"❌ Length ratio out of acceptable range: {length_ratio:.2f}")
            print("⚠️  No Level 2 reward awarded")
            print("💡 Ratio should be between 0.8-1.5x for any reward (LENIENT)")
            print("⏹️  STOPPING: Coordination requirements not met")

        if not proceed_to_level3:
            print(f"🏆 Final reward: {reward}")
            rewards.append(reward)
            continue

        # ================================================================
        # LEVEL 3: VOCABULARY DIVERSITY (UNIQUE WORDS RATIO) - STRICTER
        # ================================================================
        print("\n🔤 LEVEL 3: VOCABULARY DIVERSITY (STRICTER)")
        print("-" * 50)

        print(f"📝 Completion 1 unique words: {unique_words1}")
        print(f"📝 Completion 2 unique words: {unique_words2}")
        print(f"📊 Unique words ratio (C2/C1): {unique_words_ratio:.2f}")
        print(f"🎯 Optimal range: 0.7-1.3x (STRICTER)")
        print(f"📈 Acceptable range: 0.5-1.7x (STRICTER)")

        level3_reward = 0.0
        proceed_to_level4 = False

        if unique_words1 == 0:
            print("❌ Cannot calculate unique words ratio (no unique words in C1)")
            level3_reward = 0.0
            proceed_to_level4 = False
        elif 0.7 <= unique_words_ratio <= 1.3:
            # Perfect unique words ratio - maximum Level 3 reward (0.5)
            level3_reward = 0.5
            reward += level3_reward
            proceed_to_level4 = True
            print(
                f"✅ Excellent vocabulary diversity: +{level3_reward} (total: {reward})"
            )
            print("🎉 Optimal vocabulary balance achieved!")
            print("➡️  Proceeding to Level 4")
        elif 0.5 <= unique_words_ratio < 0.7:
            # Linear interpolation from 0 to 0.5 between 0.5 and 0.7
            level3_reward = (
                0.5 * (unique_words_ratio - 0.5) / (0.7 - 0.5)
            )  # 0.5 * (ratio - 0.5) / 0.2
            reward += level3_reward
            proceed_to_level4 = True
            print(
                f"⚠️  Below optimal vocabulary diversity: +{level3_reward:.2f} (total: {reward})"
            )
            print("💡 Consider using more diverse vocabulary in completion2")
            print("➡️  Proceeding to Level 4")
        elif 1.3 < unique_words_ratio <= 1.7:
            # Linear interpolation from 0.5 to 0 between 1.3 and 1.7
            level3_reward = 0.5 * (
                1.0 - ((unique_words_ratio - 1.3) / (1.7 - 1.3))
            )  # 0.5 * (1 - (ratio - 1.3) / 0.4)
            reward += level3_reward
            proceed_to_level4 = True
            print(
                f"⚠️  Above optimal vocabulary diversity: +{level3_reward:.2f} (total: {reward})"
            )
            print("💡 Vocabulary diversity is high but within acceptable range")
            print("➡️  Proceeding to Level 4")
        else:
            # Outside acceptable range - no Level 3 reward
            level3_reward = 0.0
            proceed_to_level4 = False
            print(
                f"❌ Vocabulary diversity out of acceptable range: {unique_words_ratio:.2f}"
            )
            print("⚠️  No Level 3 reward awarded")
            print("💡 Unique words ratio should be between 0.5-1.7x for any reward")
            print("⏹️  STOPPING: Vocabulary diversity requirements not met")

        if not proceed_to_level4:
            print(f"🏆 Final reward: {reward}")
            rewards.append(reward)
            continue

        # ================================================================
        # LEVEL 4: STYLE REWARD (TRANSITION WORDS + JACCARD SIMILARITY)
        # ================================================================
        print("\n🎨 LEVEL 4: STYLE REWARD")
        print("-" * 50)

        # Component 1: Transition Words (0.4 weight) - check in combined text (c1+c2)
        categories_found, transition_words_found = check_transition_words(c1 + c2)
        num_categories = len(categories_found)

        print(f"🔍 TRANSITION COMPONENT (40% of Level 4):")
        print(f"📊 Categories found: {num_categories}/12")
        if categories_found:
            print(f"✅ Categories: {', '.join(sorted(categories_found))}")
            print(
                f"🔤 Transition words found: {', '.join(transition_words_found[:5])}{'...' if len(transition_words_found) > 5 else ''}"
            )
        else:
            print("❌ No transition words found")

        # Calculate transition reward (0-1 scale) - NEW LOGIC
        transition_reward_scale = 0.0
        if num_categories > 0:
            # Base reward for having at least one transition word
            transition_reward_scale = 0.4
            # Additional reward for each additional category (beyond the first)
            additional_categories = num_categories - 1
            additional_reward = min(
                additional_categories * 0.05, 0.6
            )  # More conservative scaling
            transition_reward_scale += additional_reward
            transition_reward_scale = min(transition_reward_scale, 1.0)

        print(
            f"💯 Transition reward calculation: Base 0.4 + ({num_categories-1} × 0.05) = {transition_reward_scale:.3f}"
        )

        # Component 2: Jaccard Similarity (0.6 weight)
        jaccard_score_raw = calculate_jaccard_similarity(c1, c2)

        # Map Jaccard score from 0-0.23 range to 0-1 scale, with ceiling at 0.23
        jaccard_score_capped = min(jaccard_score_raw, 0.23)
        jaccard_score_normalized = jaccard_score_capped / 0.23  # Maps 0-0.23 to 0-1

        print(f"\n🔍 JACCARD SIMILARITY COMPONENT (60% of Level 4):")
        print(f"📊 Raw Jaccard similarity: {jaccard_score_raw:.4f}")
        print(f"📊 Capped Jaccard score: {jaccard_score_capped:.4f} (max: 0.23)")
        print(f"📊 Normalized Jaccard score: {jaccard_score_normalized:.4f}")
        print(f"💡 Jaccard scores 0-0.23 mapped to 0-1 scale for reward calculation")

        # Calculate final Level 4 reward
        transition_component = 0.4 * transition_reward_scale
        jaccard_component = 0.6 * jaccard_score_normalized
        level4_reward = transition_component + jaccard_component

        reward += level4_reward

        print(f"\n🎯 STYLE REWARD BREAKDOWN:")
        print(
            f"🔗 Transition component: {transition_component:.3f} (0.4 × {transition_reward_scale:.3f})"
        )
        print(
            f"📝 Jaccard component: {jaccard_component:.3f} (0.6 × {jaccard_score_normalized:.3f})"
        )
        print(f"🎨 Level 4 total: +{level4_reward:.3f} (total: {reward:.3f})")

        # Final summary
        print(f"\n🏆 REWARD BREAKDOWN:")
        print(f"📋 Level 1 (Structural): +0.5")
        print(f"⚙️  Level 2 (Coordination): +{level2_reward:.2f}")
        print(f"🔤 Level 3 (Vocabulary): +{level3_reward:.2f}")
        print(f"🎨 Level 4 (Style): +{level4_reward:.3f}")

        if level2_reward == 1.0 and level3_reward == 0.5 and level4_reward >= 0.9:
            print("\n🎉 NEAR-PERFECT ARXIV EXPANSION COORDINATION ACHIEVED!")

        print(f"🏆 FINAL REWARD: {reward:.3f} / 3.0")
        rewards.append(float(reward))

    return rewards
