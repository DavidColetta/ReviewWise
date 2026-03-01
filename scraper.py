import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

# ─────────────────────────────────────────────
# Headers — mimic a real browser to avoid blocks
# ─────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ─────────────────────────────────────────────
# URL helpers
# ─────────────────────────────────────────────

def normalize_url(url: str) -> str:
    """
    Accept either:
      - Full URL: https://www.trustpilot.com/review/dominos.com
      - Just the slug: dominos.com
    Always returns the full base URL without page number.
    """
    url = url.strip().rstrip("/")
    if url.startswith("http"):
        # Strip any existing page param
        url = re.sub(r"\?.*$", "", url)
        return url
    else:
        return f"https://www.trustpilot.com/review/{url}"


def page_url(base_url: str, page: int) -> str:
    if page == 1:
        return base_url
    return f"{base_url}?page={page}"


# ─────────────────────────────────────────────
# Parse a single page
# ─────────────────────────────────────────────

def parse_page(soup: BeautifulSoup) -> list[dict]:
    """Extract reviews from a parsed Trustpilot page."""
    reviews = []

    # Each review lives in an article tag with this data attribute
    cards = soup.find_all("article", attrs={"data-service-review-card-paper": True})

    for card in cards:
        # Review text
        body_tag = card.find("p", attrs={"data-service-review-text-typography": True})
        review_text = body_tag.get_text(strip=True) if body_tag else None
        if not review_text:
            continue  # skip reviews with no text

        # Star rating — encoded as an image alt text like "5 stars"
        rating = None
        star_img = card.find("img", alt=re.compile(r"\d star"))
        if star_img:
            match = re.search(r"(\d)", star_img["alt"])
            if match:
                rating = int(match.group(1))

        # Review title
        title_tag = card.find("h2", attrs={"data-service-review-title-typography": True})
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Date
        date_tag = card.find("time")
        date = date_tag["datetime"][:10] if date_tag and date_tag.get("datetime") else None

        reviews.append({
            "review_text": f"{title}. {review_text}".strip(". ") if title else review_text,
            "rating": rating,
            "date": date,
        })

    return reviews


# ─────────────────────────────────────────────
# Get total page count
# ─────────────────────────────────────────────

def get_total_pages(soup: BeautifulSoup) -> int:
    """Find the last page number from pagination."""
    # Trustpilot renders pagination as nav with aria-label="Pagination"
    nav = soup.find("nav", attrs={"aria-label": re.compile("pagination", re.I)})
    if not nav:
        return 1

    page_links = nav.find_all("a", href=re.compile(r"\?page=\d+"))
    page_nums = []
    for a in page_links:
        match = re.search(r"\?page=(\d+)", a["href"])
        if match:
            page_nums.append(int(match.group(1)))

    return max(page_nums) if page_nums else 1


# ─────────────────────────────────────────────
# Get business name from page
# ─────────────────────────────────────────────

def get_business_name(soup: BeautifulSoup) -> str:
    tag = soup.find("h1", class_=re.compile("title", re.I))
    if not tag:
        tag = soup.find("h1")
    if not tag:
        return "Unknown Business"

    # Only take direct text nodes — ignores child spans like "Reviews 2,423"
    from bs4 import NavigableString
    direct_text = "".join(
        str(node) for node in tag.children
        if isinstance(node, NavigableString)
    ).strip()

    # Fall back to full text if direct text is empty
    if not direct_text:
        direct_text = tag.get_text(separator=" ", strip=True)

    # Strip any trailing noise like "Reviews", digits, commas
    direct_text = re.sub(r"(Reviews?[\s\d,]*)$", "", direct_text, flags=re.I).strip()

    return direct_text or "Unknown Business"


# ─────────────────────────────────────────────
# Main scraper
# ─────────────────────────────────────────────

def scrape_trustpilot(
    url: str,
    max_pages: int = 5,
    progress_callback=None,
) -> tuple[pd.DataFrame, str]:
    """
    Scrape reviews from a Trustpilot business page.

    Args:
        url:               Full Trustpilot URL or just the slug (e.g. 'dominos.com')
        max_pages:         Max number of pages to scrape (20 reviews per page)
        progress_callback: Optional fn(current_page, total_pages) for progress bars

    Returns:
        (DataFrame with review_text, rating, date columns,  business name string)

    Raises:
        ValueError: if the URL is invalid or page can't be fetched
    """
    base_url = normalize_url(url)
    session = requests.Session()
    session.headers.update(HEADERS)

    # ── Fetch page 1 ──
    resp = session.get(page_url(base_url, 1), timeout=10)
    if resp.status_code != 200:
        raise ValueError(
            f"Could not fetch Trustpilot page (status {resp.status_code}). "
            "Check the URL and try again."
        )

    soup = BeautifulSoup(resp.text, "html.parser")
    business_name = get_business_name(soup)
    total_pages = min(get_total_pages(soup), max_pages)

    all_reviews = parse_page(soup)

    if progress_callback:
        progress_callback(1, total_pages)

    # ── Fetch remaining pages ──
    for page_num in range(2, total_pages + 1):
        time.sleep(0.8)  # polite delay between requests
        resp = session.get(page_url(base_url, page_num), timeout=10)
        if resp.status_code != 200:
            break  # stop gracefully if blocked

        page_soup = BeautifulSoup(resp.text, "html.parser")
        page_reviews = parse_page(page_soup)
        if not page_reviews:
            break  # no more reviews

        all_reviews.extend(page_reviews)

        if progress_callback:
            progress_callback(page_num, total_pages)

    if not all_reviews:
        raise ValueError(
            "No reviews found. The page structure may have changed, "
            "or the business has no reviews yet."
        )

    df = pd.DataFrame(all_reviews)
    return df, business_name


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df, name = scrape_trustpilot("https://www.trustpilot.com/review/dominos.com", max_pages=2)
    print(f"Business: {name}")
    print(f"Reviews scraped: {len(df)}")
    print(df.head())