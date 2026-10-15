import re

_ARABIC_TO_PERSIAN = str.maketrans({
    "ي": "ی",  # ي عربی -> ی فارسی
    "ك": "ک",  # ك عربی -> ک فارسی
})

_MULTI_SPACE = re.compile(r"\s+")


def normalize_persian_text(text: str) -> str:
    """آماده‌سازی حداقلی متن فارسی. نیم‌فاصله، علائم و ایموجی حفظ می‌شوند."""
    if not isinstance(text, str):
        return ""
    text = text.translate(_ARABIC_TO_PERSIAN)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()