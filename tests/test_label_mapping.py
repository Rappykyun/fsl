from fsl.constants import WORD_LABEL_TO_FILIPINO


def test_word_label_mapping_expected_values() -> None:
    assert WORD_LABEL_TO_FILIPINO["GOOD MORNING"] == "MAGANDANG UMAGA"
    assert WORD_LABEL_TO_FILIPINO["GOOD EVENING"] == "MAGANDANG GABI"
    assert WORD_LABEL_TO_FILIPINO["HELLO"] == "KAMUSTA"
    assert WORD_LABEL_TO_FILIPINO["THANK YOU"] == "SALAMAT"
    assert WORD_LABEL_TO_FILIPINO["YOURE WELCOME"] == "WALANG ANUMAN"
