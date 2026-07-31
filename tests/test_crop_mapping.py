from hardsub_ocr.video.frame_mapper import Rect, map_widget_crop, video_display_rect


def test_letterbox_mapping():
    display = video_display_rect(1000, 1000, 1920, 1080)
    assert display.y == 218.75
    crop = map_widget_crop(Rect(0, display.y, 1000, display.height), display, 1920, 1080)
    assert (crop.x, crop.y, crop.width, crop.height) == (0, 0, 1920, 1080)


def test_clips_selection_to_video():
    display = video_display_rect(1000, 1000, 1920, 1080)
    crop = map_widget_crop(Rect(-50, 0, 1100, 1000), display, 1920, 1080)
    assert crop.width == 1920 and crop.height == 1080

