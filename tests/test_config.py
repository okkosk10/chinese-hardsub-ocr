import pytest
from hardsub_ocr.config import Crop


def test_crop_parse(): assert Crop.parse("400, 700,1120,180") == Crop(400, 700, 1120, 180)


def test_crop_invalid():
    with pytest.raises(ValueError): Crop.parse("1,2,3")

