import csv
import random
import re
import time
from html import unescape
from pathlib import Path
from typing import Any, Dict, List

import requests


GTOKEN = ""

BASE_URL = "https://www.mfa.gov.by"
PAGE_URL = f"{BASE_URL}/press/news_mfa/"
API_URL = f"{BASE_URL}/press/news_mfa/json/?get_more_news"
OUTPUT_FILE = Path("mfa_news.csv")
MAX_PAGES = 5000
RETRIES = 3
LIST_DELAY_MIN = 0.25
LIST_DELAY_MAX = 0.45
NO_PROGRESS_PAGE_LIMIT = 30
CONTENT_DELAY_MIN = 0.6
CONTENT_DELAY_MAX = 1.1
SAVE_EVERY = 20
MAX_CONSECUTIVE_CONTENT_ERRORS = 15
FIELDNAMES = ["num", "date", "year", "title", "url", "tag", "speaker", "content"]


def get_year(date_value: str) -> str:
    if not date_value:
        return ""
    for item in reversed(re.findall(r"\d{4}", str(date_value))):
        return item
    return ""


def clean_date(date_value: str) -> str:
    return " ".join(str(date_value or "").replace(",", " ").split())


def html_to_text(html: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    text = re.sub(r"</p\s*>", "\n\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = unescape(text).replace("\xa0", " ")
    text = re.sub(r"\r", "", text)
    lines = [line.strip() for line in text.splitlines()]

    chunks = []
    buffer = []
    for line in lines:
        if line:
            buffer.append(line)
        elif buffer:
            chunks.append(" ".join(buffer))
            buffer = []

    if buffer:
        chunks.append(" ".join(buffer))

    return "\n\n".join(chunks).strip()


def extract_content(html: str) -> str:
    match = re.search(
        r'<span class="date _big">.*?</span>(.*?)(?:<p>\s*<a href="/print/press/news_mfa/|<div class="content-page__social")',
        html,
        flags=re.S | re.I,
    )
    if not match:
        return ""
    return html_to_text(match.group(1))


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": PAGE_URL,
        }
    )
    return session


def get_gtoken(session: requests.Session) -> str:
    if GTOKEN:
        return GTOKEN

    for attempt in range(RETRIES):
        try:
            response = session.get(PAGE_URL, timeout=20)
            response.encoding = "windows-1251"
            response.raise_for_status()
            match = re.search(r'id="ggtoken"[^>]*value="([^"]+)"', response.text)
            if match:
                return match.group(1)
            return ""
        except requests.RequestException:
            if attempt < RETRIES - 1:
                time.sleep(1)
    return ""


def load_json_page(session: requests.Session, page: int, gtoken: str) -> List[Dict[str, Any]]:
    params = {
        "lang": "ru",
        "tid": "ru_news_mfa",
        "page": page,
        "ggtoken": gtoken,
    }

    for attempt in range(RETRIES):
        try:
            response = session.get(API_URL, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            news = data.get("news", [])
            return news if isinstance(news, list) else []
        except requests.RequestException:
            if attempt < RETRIES - 1:
                time.sleep(1.5)
        except ValueError:
            return []
    return []


def load_content(session: requests.Session, url: str) -> str:
    if not url:
        return ""

    for attempt in range(RETRIES):
        try:
            response = session.get(url, timeout=20)
            response.encoding = "windows-1251"
            response.raise_for_status()
            return extract_content(response.text)
        except requests.RequestException:
            if attempt < RETRIES - 1:
                time.sleep(2)
        except Exception:
            return ""
    return ""


def read_existing_rows() -> List[Dict[str, str]]:
    if not OUTPUT_FILE.exists():
        return []

    for delimiter in (";", ","):
        try:
            with OUTPUT_FILE.open("r", newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                if reader.fieldnames and "url" in reader.fieldnames:
                    rows = []
                    for row in reader:
                        rows.append({field: str(row.get(field, "") or "") for field in FIELDNAMES})
                    return rows
        except Exception:
            continue

    return []


def write_rows(rows: List[Dict[str, str]]) -> None:
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES,
            delimiter=";",
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)


def build_or_resume_index(session: requests.Session, rows: List[Dict[str, str]], gtoken: str) -> List[Dict[str, str]]:
    by_url = {row["url"]: row for row in rows if row.get("url")}
    total = len(rows)
    stop_page = None
    no_progress_pages = 0
    seen_signatures = set()
    min_year_seen = min((row["year"] for row in rows if row.get("year")), default="")

    for page in range(1, MAX_PAGES + 1):
        items = load_json_page(session, page, gtoken)
        count = len(items)

        if count == 0:
            print(f"page {page} | items 0 | total {len(by_url)}")
            stop_page = page
            break

        added = 0
        page_urls = []
        for item in items:
            relative_url = str(item.get("url", "") or "")
            full_url = f"{BASE_URL}{relative_url}" if relative_url.startswith("/") else relative_url
            if full_url:
                page_urls.append(full_url)
            if not full_url or full_url in by_url:
                continue

            date_value = clean_date(item.get("date", ""))
            total += 1
            row = {
                "num": str(total),
                "date": date_value,
                "year": get_year(date_value),
                "title": str(item.get("title", "") or "").strip(),
                "url": full_url,
                "tag": "",
                "speaker": "",
                "content": "",
            }
            rows.append(row)
            by_url[full_url] = row
            added += 1

        min_year = min((row["year"] for row in rows if row.get("year")), default="")
        signature = tuple(page_urls)
        signature_seen_before = signature in seen_signatures
        seen_signatures.add(signature)
        year_progressed = bool(min_year) and (not min_year_seen or min_year < min_year_seen)
        if year_progressed:
            min_year_seen = min_year

        if added == 0 and not year_progressed and signature_seen_before:
            no_progress_pages += 1
        else:
            no_progress_pages = 0

        print(
            f"page {page} | items {count} | added {added} | total {len(by_url)} | "
            f"min_year {min_year} | no_progress {no_progress_pages}",
        )

        if added:
            write_rows(rows)

        if min_year == "2000" and no_progress_pages >= NO_PROGRESS_PAGE_LIMIT:
            print(
                f"stop: reached lower bound year 2000 and saw {NO_PROGRESS_PAGE_LIMIT} "
                "no-progress pages in a row",
            )
            stop_page = page
            break

        time.sleep(random.uniform(LIST_DELAY_MIN, LIST_DELAY_MAX))

    if stop_page is None:
        write_rows(rows)

    return rows


def enrich_content(session: requests.Session, rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    pending = [row for row in rows if not row.get("content", "").strip()]
    total_pending = len(pending)

    if not total_pending:
        print("content 0/0 | done")
        return rows

    done = 0
    consecutive_errors = 0

    for row in pending:
        content = load_content(session, row["url"])

        if content:
            row["content"] = content
            consecutive_errors = 0
        else:
            consecutive_errors += 1

        done += 1
        print(f"content {done}/{total_pending} | saved {sum(1 for r in rows if r.get('content', '').strip())}")

        if done % SAVE_EVERY == 0 or content:
            write_rows(rows)

        if consecutive_errors >= MAX_CONSECUTIVE_CONTENT_ERRORS:
            write_rows(rows)
            print("too many content errors, stop and run again later")
            return rows

        time.sleep(random.uniform(CONTENT_DELAY_MIN, CONTENT_DELAY_MAX))

    write_rows(rows)
    return rows


def main() -> None:
    session = create_session()
    gtoken = get_gtoken(session)

    rows = read_existing_rows()
    rows = build_or_resume_index(session, rows, gtoken)
    write_rows(rows)


if __name__ == "__main__":
    main()
