from hardsub_ocr.subtitle.segment_builder import SegmentBuilder


def test_segment_start_extend_change_finish():
    b = SegmentBuilder(blank_tolerance=0)
    assert b.add(1.0, "你好世界", .8, 0)[0] == "start"
    assert b.add(1.5, "你好世界", .9, 1)[0] == "extend"
    assert b.add(2.0, "再见朋友", .8, 2)[0] == "replace"
    segments = b.finish(2.5)
    assert len(segments) == 2 and segments[0].end_time <= segments[1].start_time


def test_blank_and_min_duration():
    b = SegmentBuilder(blank_tolerance=0, min_duration=.4)
    b.add(1.0, "好", .9, 0); b.add(1.1, "", 0, 1)
    assert b.segments[0].end_time == 1.4


def test_interrupted_finish():
    b = SegmentBuilder(); b.add(3, "字幕", .8, 0)
    assert b.finish(3.2)[0].text == "字幕"


def test_rapid_changes_never_overlap():
    b = SegmentBuilder(min_duration=.5)
    b.add(0, "第一句", .9, 0)
    b.add(.1, "完全不同", .9, 1)
    segments = b.finish(.3)
    assert segments[0].end_time <= segments[1].start_time
