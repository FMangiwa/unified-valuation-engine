import requests, config
from pydantic import BaseModel
from engine.scraper import DealScraper
from engine.pricer_qlora import QLoRAPricer

class DealOpportunity(BaseModel):
    title: str
    listed_price: float
    estimated_value: float
    margin_discount_pct: float
    confidence: str
    url: str = ""

class PushoverMessenger:
    def __init__(self):
        self.user_key=config.PUSHOVER_USER_KEY
        self.api_token=config.PUSHOVER_API_TOKEN
    def send_notification(self,title,message,url=""):
        if not self.user_key or not self.api_token: return False
        data={"token":self.api_token,"user":self.user_key,"title":title,"message":message,"priority":1}
        if url: data.update({"url":url,"url_title":"View Deal Listing"})
        try: return requests.post("https://api.pushover.net/1/messages.json",data=data,timeout=5).status_code==200
        except requests.RequestException: return False

class AutonomousPlanningAgent:
    def __init__(self):
        self.scraper=DealScraper()
        self.pricer=QLoRAPricer()
        self.notifier=PushoverMessenger()
        self.memory_urls=[]

    def run_deal_discovery_cycle(self):
        items=self.scraper.fetch_latest_listings(self.memory_urls)
        out=[]
        for item in items:
            if item.get("url"): self.memory_urls.append(item["url"])
            e=self.pricer.predict_all(item["description"])
            est=e["ensemble_est"]
            if est<=0 or item["listed_price"]<=0: continue
            discount=round((est-item["listed_price"])/est*100,1)
            if item["listed_price"] <= est*config.DEAL_THRESHOLD:
                deal=DealOpportunity(
                    title=item["description"][:80],
                    listed_price=item["listed_price"],
                    estimated_value=est,
                    margin_discount_pct=discount,
                    confidence=e["confidence"],
                    url=item.get("url","")
                )
                out.append(deal)
                self.notifier.send_notification(
                    "🎯 High-Margin Deal",
                    f"Listed: ${item['listed_price']:.2f} | Fair value: ${est:.2f} | Discount: {discount}% | Confidence: {e['confidence']}",
                    item.get("url","")
                )
        return out
