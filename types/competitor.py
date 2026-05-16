from pydantic import BaseModel, Field
from typing import Optional


class CompetitorBasicInfo(BaseModel):
    """Structured competitor basic info extracted by LLM."""
    company_name: str = ""
    founded: Optional[str] = None
    founders: list[str] = Field(default_factory=list)
    ceo: str = ""
    headquarters: str = ""
    employee_count: Optional[str] = None
    funding_stage: str = ""
    funding_amount: str = ""
    description: str = ""


class ProductFeature(BaseModel):
    """A single product feature with score."""
    name: str
    description: str = ""
    score: int = 0  # 1-5
    category: str = ""  # e.g. "core", "ux", "integration", "pricing"


class MarketMetric(BaseModel):
    """Market performance metrics."""
    user_count: Optional[str] = None
    market_share: Optional[str] = None
    revenue: Optional[str] = None
    growth_rate: Optional[str] = None
    funding_total: Optional[str] = None


class CompetitorProfile(BaseModel):
    """Complete competitor profile from collection phase."""
    competitor: str
    basic_info: CompetitorBasicInfo = Field(default_factory=CompetitorBasicInfo)
    features: list[ProductFeature] = Field(default_factory=list)
    market: MarketMetric = Field(default_factory=MarketMetric)
    user_reviews_summary: str = ""
    strategic_news: list[str] = Field(default_factory=list)
    collected_at: str = ""
