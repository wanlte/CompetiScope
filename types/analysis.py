from pydantic import BaseModel, Field


class SWOTResult(BaseModel):
    """Structured SWOT analysis for a single competitor."""
    competitor: str
    strengths: list[str] = Field(default_factory=list, min_length=1)
    weaknesses: list[str] = Field(default_factory=list, min_length=1)
    opportunities: list[str] = Field(default_factory=list, min_length=1)
    threats: list[str] = Field(default_factory=list, min_length=1)


class FeatureScore(BaseModel):
    """Feature comparison score for one dimension."""
    feature_name: str
    description: str = ""
    competitor_scores: dict[str, int] = Field(default_factory=dict)  # competitor_name -> score (1-5)
    our_score: int = 0


class CompetitiveLandscape(BaseModel):
    """Competitive landscape analysis."""
    market_positions: dict[str, str] = Field(default_factory=dict)
    competitive_intensity: str = ""
    differentiation_factors: list[str] = Field(default_factory=list)
    market_gaps: list[str] = Field(default_factory=list)
    leader: str = ""
    challengers: list[str] = Field(default_factory=list)
    niche_players: list[str] = Field(default_factory=list)


class KeyInsights(BaseModel):
    """Extracted key insights from analysis."""
    insights: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    opportunities: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """Complete analysis result."""
    competitors: list[str]
    feature_comparison: list[FeatureScore] = Field(default_factory=list)
    swot_analysis: list[SWOTResult] = Field(default_factory=list)
    competitive_landscape: CompetitiveLandscape = Field(default_factory=CompetitiveLandscape)
    key_insights: KeyInsights = Field(default_factory=KeyInsights)
