import logging, time, feedparser, requests
from bs4 import BeautifulSoup
from openai import OpenAI
from typing import List, Optional
from pydantic import BaseModel, Field
import config
import prompt_config

logger = logging.getLogger("DealScraper")
logger.setLevel(logging.INFO)

class ScrapedItem(BaseModel):
    category: str
    condition: str
    title: str
    features: List[str] = Field(default_factory=list)
    listed_price: float
    url: str

class DealSelection(BaseModel):
    deals: List[ScrapedItem]

class RawRSSDeal:
    DEFAULT_FEEDS = [
        "https://www.dealnews.com/rss/c142/",
        "https://www.dealnews.com/rss/c39/",
        "https://www.dealnews.com/rss/c196/",
    ]

    def __init__(self, entry):
        self.title = entry.get("title", "")
        self.summary = entry.get("summary", "")
        self.url = entry.get("link", "")
        self.page_text = ""

    def fetch_page_content(self):
        if not self.url:
            return
        try:
            headers = {"User-Agent": "Mozilla/5.0 DealEngine/2.0"}
            res = requests.get(self.url, headers=headers, timeout=8)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, "html.parser")
                self.page_text = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p")[:8])
        except Exception as e:
            logger.debug("Page scrape failed: %s", e)

    def describe(self):
        text = f"Title: {self.title}\nSummary: {self.summary}\nURL: {self.url}"
        if self.page_text:
            text += f"\nDetailed Page Text: {self.page_text[:1000]}"
        return text

    @classmethod
    def fetch_all_raw(cls, feed_urls=None, limit_per_feed=10):
        deals = []
        for feed_url in feed_urls or cls.DEFAULT_FEEDS:
            try:
                parsed = feedparser.parse(feed_url)
                for entry in parsed.entries[:limit_per_feed]:
                    d = cls(entry)
                    d.fetch_page_content()
                    deals.append(d)
                    time.sleep(0.05)
            except Exception as e:
                logger.error("Feed error %s: %s", feed_url, e)
        return deals

class DealScraper:
    MODEL = config.FRONTIER_OPENAI_MODEL

    SYSTEM_PROMPT = """Extract product identity and technical metadata from deal listings.
Return only products with explicit prices and clear specifications.
Condition must be New or Refurbished. Do not confuse capacity, screen size,
model numbers, RAM, storage, wattage, etc. with monetary price."""

    def __init__(self):
        self.openai = OpenAI(api_key=config.OPENAI_API_KEY)

    def fetch_latest_listings(self, memory_urls=None):
        memory_urls = set(memory_urls or [])
        raw = [x for x in RawRSSDeal.fetch_all_raw() if x.url not in memory_urls]
        if not raw:
            return []

        body = "\n\n".join(x.describe() for x in raw)
        prompt = (
            "Select up to 5 promising deals with explicit price and specifications.\n"
            "Do not infer a price that is not present.\n\n" + body
        )
        try:
            response = self.openai.chat.completions.parse(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format=DealSelection,
            )
            results = []
            for item in response.choices[0].message.parsed.deals:
                if item.listed_price <= 0:
                    continue
                formatted = prompt_config.format_input(
                    item.category, item.condition, item.title, item.features
                )
                results.append({
                    "title": item.title,
                    "description": formatted,
                    "instruction": prompt_config.INSTRUCTION,
                    "input": formatted,
                    "listed_price": float(item.listed_price),
                    "url": item.url,
                })
            return results
        except Exception as e:
            logger.error("Structured extraction failed: %s", e)
            return []
