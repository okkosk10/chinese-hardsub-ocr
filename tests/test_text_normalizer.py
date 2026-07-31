from hardsub_ocr.subtitle.text_normalizer import normalize_text


def test_chinese_spacing_and_punctuation(): assert normalize_text(" 你好 ，  世界！！ ") == "你好，世界！"


def test_lines_and_duplicate(): assert normalize_text("第一行\r\n第一行\n 第二行 ") == "第一行\n第二行"

