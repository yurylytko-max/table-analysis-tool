from mfa_news_en import create_session, enrich_content, read_existing_rows, write_rows


def main() -> None:
    rows = read_existing_rows()
    if not rows:
        raise SystemExit("mfa_news_en.csv is empty or missing.")

    session = create_session()
    rows = enrich_content(session, rows)
    write_rows(rows)


if __name__ == "__main__":
    main()
