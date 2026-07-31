from hardsub_ocr.subtitle.text_similarity import is_same_text, similarity


def test_spacing_ignored(): assert similarity("你好 世界", "你好世界") == 100


def test_short_text_strict(): assert not is_same_text("你", "他", 70, 94)

