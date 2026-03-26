import csv
import html
import re
from pathlib import Path


SOURCE_FILE = Path("mfa_news.csv")
OUTPUT_FILE = Path("mfa_news_clean.csv")
FIELDNAMES = ["num", "date", "year", "title", "url", "tag", "speaker", "content"]


def clean_entities(text: str) -> str:
    value = html.unescape(text or "")
    value = value.replace("\xa0", " ")
    return value


def strip_js_blocks(text: str) -> str:
    patterns = [
        r"\(function\(\$\)\s*\{.*$",
        r"\$\('#ad-gallery_cke_.*?(?=(?:\n\s*\n)|$)",
        r"IMAGE_LINK_[A-Z_]+.*?(?=(?:\n\s*\n)|$)",
        r"\$\.fancybox\.open\(.*?(?=(?:\n\s*\n)|$)",
        r"addGallery\(.*?(?=(?:\n\s*\n)|$)",
        r"/ckeditor/plugins/slideshow/.*?(?=(?:\n\s*\n)|$)",
        r"window\.open\(idesc.*?(?=(?:\n\s*\n)|$)",
        r"\}\)\(jQuery\);.*$",
    ]

    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.S)
    return cleaned


def clean_text(text: str) -> str:
    value = clean_entities(text)
    value = strip_js_blocks(value)
    value = re.sub(r"&[#A-Za-z0-9]+;", " ", value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = re.sub(r"\n[ \t]+", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    value = re.sub(r"[ \t]{2,}", " ", value)
    return value.strip()


def main() -> None:
    with SOURCE_FILE.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src, delimiter=";")
        rows = []

        for row in reader:
            cleaned_row = {field: row.get(field, "") for field in FIELDNAMES}
            cleaned_row["title"] = clean_text(cleaned_row["title"])
            cleaned_row["content"] = clean_text(cleaned_row["content"])
            rows.append(cleaned_row)

    with OUTPUT_FILE.open("w", encoding="utf-8-sig", newline="") as dst:
        writer = csv.DictWriter(
            dst,
            fieldnames=FIELDNAMES,
            delimiter=";",
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
