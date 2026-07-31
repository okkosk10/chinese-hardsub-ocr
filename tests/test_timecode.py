import pytest
from hardsub_ocr.utils.timecode import format_timecode, parse_timecode


@pytest.mark.parametrize(("value", "expected"), [("00:04:30", 270), ("00:04:30.500", 270.5), ("12.25", 12.25)])
def test_parse_timecode(value, expected): assert parse_timecode(value) == expected


def test_invalid_timecode():
    with pytest.raises(ValueError): parse_timecode("00:99:00")


def test_srt_format(): assert format_timecode(270.5, True) == "00:04:30,500"

