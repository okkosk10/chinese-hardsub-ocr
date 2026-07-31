from hardsub_ocr.cli import clean_srt_main
from hardsub_ocr.models import SubtitleSegment
from hardsub_ocr.subtitle.srt_cleaner import clean_segments
from hardsub_ocr.subtitle.text_similarity import line_order_invariant_similarity, similarity


def segment(index, start, end, text, confidence=.9):
    return SubtitleSegment(index, start, end, text, confidence)


def test_line_order_swap_is_same_subtitle():
    left = "你先回去\n我随后就来"
    right = "我随后就来\n你先回去"
    assert line_order_invariant_similarity(left, right) == 100
    assert similarity(left, right) == 100


def test_short_ascii_number_symbol_noise_is_removed():
    cleaned, removed = clean_segments([
        segment(1, 0, .7, "A1", .95),
        segment(2, 1, 1.8, "#$", .9),
    ])
    assert cleaned == []
    assert {item["reason"] for item in removed} == {"standalone_short_non_chinese"}


def test_transient_low_confidence_non_chinese_noise_is_removed():
    cleaned, removed = clean_segments([segment(1, 0, .8, "LOGO", .45)])
    assert cleaned == []
    assert removed[0]["reason"] == "transient_non_chinese_low_confidence"


def test_normal_short_chinese_dialogue_is_preserved():
    source = [segment(index, index, index + .4, text, .45)
              for index, text in enumerate(("爸爸", "好的", "不要", "谢谢"), 1)]
    cleaned, removed = clean_segments(source)
    assert [item.text for item in cleaned] == ["爸爸", "好的", "不要", "谢谢"]
    assert removed == []


def test_swapped_two_line_duplicate_is_merged():
    cleaned, removed = clean_segments([
        segment(1, 0, 1, "你先回去\n我随后就来"),
        segment(2, 1.1, 2, "我随后就来\n你先回去"),
    ])
    assert len(cleaned) == 1 and cleaned[0].end_time == 2
    assert removed[0]["reason"] == "merged_line_order_duplicate"


def test_short_fragment_near_long_chinese_is_removed():
    cleaned, removed = clean_segments([
        segment(1, 0, 2, "今天必须陪客户一起吃饭", .92),
        segment(2, 2.1, 2.5, "陪客户", .6),
    ])
    assert [item.text for item in cleaned] == ["今天必须陪客户一起吃饭"]
    assert removed[0]["reason"] == "transient_noise_near_similar_long_chinese"


def test_clean_srt_command_keeps_original_and_writes_report(tmp_path):
    source = tmp_path / "sample.srt"
    original = "1\n00:00:00,000 --> 00:00:00,700\nA1\n\n2\n00:00:01,000 --> 00:00:02,000\n谢谢\n"
    source.write_text(original, encoding="utf-8")
    assert clean_srt_main(["--input", str(source)]) == 0
    cleaned = tmp_path / "sample.cleaned.srt"
    report = tmp_path / "sample.cleaned.json"
    assert source.read_text(encoding="utf-8") == original
    assert "谢谢" in cleaned.read_text(encoding="utf-8") and "A1" not in cleaned.read_text(encoding="utf-8")
    assert report.exists()
