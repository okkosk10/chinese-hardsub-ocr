from hardsub_ocr.models import SubtitleSegment
from hardsub_ocr.subtitle.srt_writer import render_srt


def test_srt_sequence_and_no_overlap():
    segments = [SubtitleSegment(9, 1, 2, "第一", .9), SubtitleSegment(10, 1.5, 3, "第二", .8)]
    text = render_srt(segments)
    assert "1\n00:00:01,000 --> 00:00:02,000" in text
    assert "2\n00:00:02,000 --> 00:00:03,000" in text


def test_empty_omitted(): assert render_srt([SubtitleSegment(1, 0, 1, " ", 0)]) == ""

