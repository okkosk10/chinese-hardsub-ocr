from __future__ import annotations

from dataclasses import dataclass
import re

from hardsub_ocr.models import OcrCandidate
from hardsub_ocr.subtitle.text_normalizer import comparison_key
from hardsub_ocr.subtitle.text_similarity import similarity

_CHINESE_RE = re.compile(r"[\u3400-\u9fff]")
_TRAILING_PUNCT = "，。！？；：、,.!?;:"


@dataclass(frozen=True, slots=True)
class CandidateWeights:
    confidence: float = 35.0
    consensus: float = 35.0
    chinese_ratio: float = 15.0
    length_stability: float = 10.0
    transition_mix_penalty: float = 35.0
    singleton_penalty: float = 8.0


@dataclass(slots=True)
class CandidateSelection:
    selected: OcrCandidate | None
    rejected: list[OcrCandidate]
    reason: str
    consensus_score: float
    removed_unstable_suffix: str = ""
    confirmed: bool = False


def chinese_character_ratio(text: str) -> float:
    visible = [char for char in text if not char.isspace() and char not in _TRAILING_PUNCT]
    return sum(bool(_CHINESE_RE.fullmatch(char)) for char in visible) / len(visible) if visible else 0.0


def detect_transition_mix(previous_text: str, current_text: str) -> tuple[bool, str, str]:
    if previous_text and current_text.startswith(previous_text) and len(current_text) > len(previous_text):
        return True, previous_text, current_text[len(previous_text):]
    previous, current = comparison_key(previous_text), comparison_key(current_text)
    if not previous or not current or current == previous:
        return False, "", ""
    # Full previous subtitle followed by new text is the strongest transition-mix signal.
    if current.startswith(previous) and len(current) > len(previous):
        return True, previous_text, current_text[len(previous_text):]
    limit = min(12, len(previous), len(current) - 1)
    for size in range(limit, 2, -1):
        fragment = previous[-size:]
        if current.startswith(fragment) and len(current) > size:
            return True, fragment, current[size:]
    return False, "", ""


def _common_prefix(left: str, right: str) -> str:
    size = 0
    for a, b in zip(left, right):
        if a != b:
            break
        size += 1
    return left[:size]


def select_candidate(candidates: list[OcrCandidate], previous_text: str = "", consensus_threshold: float = 88.0,
                     remove_unstable_suffix: bool = True, unstable_suffix_max_chars: int = 3,
                     weights: CandidateWeights = CandidateWeights()) -> CandidateSelection:
    if not candidates:
        return CandidateSelection(None, [], "no_candidates", 0.0, confirmed=False)
    nonempty = [candidate for candidate in candidates if comparison_key(candidate.normalized_text)]
    if not nonempty:
        return CandidateSelection(None, candidates, "all_candidates_empty", 0.0, confirmed=False)
    lengths = sorted(len(comparison_key(candidate.normalized_text)) for candidate in nonempty)
    median_length = lengths[len(lengths) // 2]
    for candidate in nonempty:
        others = [similarity(candidate.normalized_text, other.normalized_text) for other in nonempty if other is not candidate]
        candidate.similarity_to_other_candidates = sum(others) / len(others) if others else 100.0
        candidate.previous_text_similarity = similarity(previous_text, candidate.normalized_text) if previous_text else 0.0
        candidate.character_count = len(comparison_key(candidate.normalized_text))
        candidate.chinese_character_ratio = chinese_character_ratio(candidate.normalized_text)
        mixed, matched, remaining = detect_transition_mix(previous_text, candidate.normalized_text)
        candidate.transition_mix_detected = mixed
        candidate.matched_previous_fragment, candidate.remaining_new_fragment = matched, remaining
        length_stability = 1.0 - min(1.0, abs(candidate.character_count - median_length) / max(1, median_length))
        agreeing = sum(similarity(candidate.normalized_text, other.normalized_text) >= consensus_threshold
                       for other in nonempty if other is not candidate)
        candidate.candidate_score = (
            candidate.confidence * weights.confidence
            + candidate.similarity_to_other_candidates / 100 * weights.consensus
            + candidate.chinese_character_ratio * weights.chinese_ratio
            + length_stability * weights.length_stability
            - (weights.transition_mix_penalty if mixed else 0)
            - (weights.singleton_penalty if len(nonempty) > 1 and agreeing == 0 else 0)
        )

    ranked = sorted(nonempty, key=lambda item: (item.candidate_score, item.confidence), reverse=True)
    selected = ranked[0]
    agreeing = [item for item in nonempty if similarity(selected.normalized_text, item.normalized_text) >= consensus_threshold]
    reason = "highest_weighted_score"
    removed_suffix = ""

    # Prefer a repeated exact/near-exact consensus over a unique longer suffix.
    clusters: list[list[OcrCandidate]] = []
    for candidate in nonempty:
        placed = False
        for cluster in clusters:
            if similarity(candidate.normalized_text, cluster[0].normalized_text) >= consensus_threshold:
                cluster.append(candidate); placed = True; break
        if not placed:
            clusters.append([candidate])
    clusters.sort(key=lambda cluster: (len(cluster), max(x.character_count for x in cluster),
                                       max(x.confidence for x in cluster)), reverse=True)
    if len(clusters[0]) >= 2:
        cluster = clusters[0]
        # Most frequently observed normalized string wins; confidence breaks ties.
        selected = max(cluster, key=lambda item: (
            sum(comparison_key(x.normalized_text) == comparison_key(item.normalized_text) for x in cluster),
            item.confidence,
        ))
        agreeing = cluster
        reason = "candidate_consensus"

    if remove_unstable_suffix and len(nonempty) >= 2:
        selected_key = comparison_key(selected.normalized_text)
        shorter_groups: dict[str, list[OcrCandidate]] = {}
        for item in nonempty:
            key = comparison_key(item.normalized_text)
            shorter_groups.setdefault(key, []).append(item)
        stable_shorter = [group for key, group in shorter_groups.items() if len(group) >= 2 and selected_key.startswith(key)
                          and 0 < len(selected_key) - len(key) <= unstable_suffix_max_chars]
        if stable_shorter:
            stable = max(stable_shorter, key=len)
            short = max(stable, key=lambda item: item.confidence)
            removed_suffix = selected.normalized_text[len(_common_prefix(selected.normalized_text, short.normalized_text)):]
            selected, agreeing = short, stable
            reason = "removed_unique_unstable_suffix"

        # The consensus may already have selected the shorter stable sentence.
        # Still record a suffix observed only on one longer outlier.
        selected_key = comparison_key(selected.normalized_text)
        if not removed_suffix:
            outliers = [item for item in nonempty
                        if comparison_key(item.normalized_text).startswith(selected_key)
                        and 0 < len(comparison_key(item.normalized_text)) - len(selected_key) <= unstable_suffix_max_chars
                        and sum(comparison_key(other.normalized_text) == comparison_key(item.normalized_text)
                                for other in nonempty) == 1]
            if outliers and len(agreeing) >= 2:
                outlier = max(outliers, key=lambda item: item.character_count)
                prefix = _common_prefix(outlier.normalized_text, selected.normalized_text)
                removed_suffix = outlier.normalized_text[len(prefix):]
                reason = "candidate_consensus_removed_unstable_suffix"

        # Fast mode may provide only two observations. If they differ solely by
        # a very short suffix and confidence is comparable, prefer the observed
        # common sentence instead of trusting the unique extension.
        if not removed_suffix and len(nonempty) == 2:
            ordered = sorted(nonempty, key=lambda item: item.character_count)
            short, long = ordered[0], ordered[1]
            short_key, long_key = comparison_key(short.normalized_text), comparison_key(long.normalized_text)
            if (long_key.startswith(short_key)
                    and 0 < len(long_key) - len(short_key) <= unstable_suffix_max_chars
                    and short.confidence + 0.12 >= long.confidence):
                prefix = _common_prefix(long.normalized_text, short.normalized_text)
                removed_suffix = long.normalized_text[len(prefix):]
                selected, agreeing = short, [short]
                reason = "removed_two_frame_unstable_suffix"

    consensus = sum(similarity(selected.normalized_text, item.normalized_text) for item in agreeing) / len(agreeing)
    confirmed = len(agreeing) >= 2 or selected.confidence >= 0.75
    rejected = [candidate for candidate in candidates if candidate is not selected]
    return CandidateSelection(selected, rejected, reason, consensus, removed_suffix, confirmed)
