"""Shared constants and label mappings."""

WORD_IDS = [0, 2, 3, 7, 8]

WORD_ID_TO_LABEL = {
    0: "GOOD MORNING",
    2: "GOOD EVENING",
    3: "HELLO",
    7: "THANK YOU",
    8: "YOURE WELCOME",
}

WORD_LABEL_TO_FILIPINO = {
    "GOOD MORNING": "MAGANDANG UMAGA",
    "GOOD EVENING": "MAGANDANG GABI",
    "HELLO": "KAMUSTA",
    "THANK YOU": "SALAMAT",
    "YOURE WELCOME": "WALANG ANUMAN",
}

LETTER_CLASSES = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
