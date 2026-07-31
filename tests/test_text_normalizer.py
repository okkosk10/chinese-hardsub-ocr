from hardsub_ocr.subtitle.text_normalizer import merge_ocr_lines, normalize_text


def test_chinese_spacing_and_punctuation(): assert normalize_text(" 你好 ，  世界！！ ") == "你好，世界！"


def test_lines_and_duplicate(): assert normalize_text("第一行\r\n第一行\n 第二行 ") == "第一行\n第二行"


def test_merge_one_character_boundary_overlap():
    assert merge_ocr_lines(["结衣，真", "真不好意思，让你送我"])[1] == "结衣，真不好意思，让你送我"


def test_merge_multi_character_boundary_overlap():
    assert merge_ocr_lines(["你要照顾", "照顾好自己哦"])[1] == "你要照顾好自己哦"


def test_single_line_is_unchanged():
    assert merge_ocr_lines(["晚饭呢？"])[1] == "晚饭呢？"


def test_url_and_number_boundaries_are_not_deduplicated():
    assert merge_ocr_lines(["https://a1", "1.example"])[1] == "https://a11.example"
    assert merge_ocr_lines(["订单123", "123完成"])[1] == "订单123123完成"


def test_short_real_one_character_repeat_is_preserved():
    assert merge_ocr_lines(["好", "好看"])[1] == "好好看"
