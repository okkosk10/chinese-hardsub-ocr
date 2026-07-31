from hardsub_ocr.models import OcrCandidate
from hardsub_ocr.subtitle.candidate_selector import select_candidate


def candidate(text: str, confidence: float = .85, timestamp: float = 1.0) -> OcrCandidate:
    return OcrCandidate(timestamp, text, text, confidence, "gray2x")


def test_consensus_removes_unique_suspicious_suffix():
    selection = select_candidate([
        candidate("是的，要陪客户，抱啊", .92),
        candidate("是的，要陪客户", .86, 1.2),
        candidate("是的，要陪客户", .84, 1.4),
    ])
    assert selection.selected.normalized_text == "是的，要陪客户"
    assert selection.reason in {"candidate_consensus_removed_unstable_suffix", "removed_unique_unstable_suffix"}
    assert selection.removed_unstable_suffix == "，抱啊"


def test_repeated_complete_sentence_beats_short_frame():
    selection = select_candidate([
        candidate("结衣，真", .93),
        candidate("结衣，真不好意思，让你送我", .84, 1.2),
        candidate("结衣，真不好意思，让你送我", .82, 1.4),
    ])
    assert selection.selected.normalized_text == "结衣，真不好意思，让你送我"


def test_transition_mix_is_rejected_by_new_text_consensus():
    selection = select_candidate([
        candidate("又去打高尔夫吗？是的", .95),
        candidate("是的，要陪客户", .83, 1.2),
        candidate("是的，要陪客户", .81, 1.4),
    ], previous_text="又去打高尔夫吗？")
    assert selection.selected.normalized_text == "是的，要陪客户"
    mixed = next(item for item in selection.rejected if item.normalized_text.startswith("又去"))
    assert mixed.transition_mix_detected


def test_fast_two_frame_suffix_prefers_common_text():
    selection = select_candidate([
        candidate("是的，要陪客户，抱啊", .91),
        candidate("是的，要陪客户", .84, 1.2),
    ])
    assert selection.selected.normalized_text == "是的，要陪客户"
    assert selection.removed_unstable_suffix == "，抱啊"
