import os
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import yfinance as yf
from google import genai
from google.genai import types

# Initialize app
app = FastAPI(title="Earnings Prediction Agent", docs_url="/api/docs", openapi_url="/api/openapi.json")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini Client
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "EarningsResearchApp user@example.com")

class PredictionRequest(BaseModel):
    ticker: str
    cutoff_date: Optional[str] = None  # Format: YYYY-MM-DD

class PredictionResponse(BaseModel):
    ticker: str
    composite_score: float = Field(..., ge=-1.0, le=1.0)
    verdict: str
    expected_gap_magnitude_pct: float
    priced_in_implied_move_pct: float
    is_edge_present: bool
    summary: str
    key_drivers: List[str]
    tail_risks: List[str]

def calculate_implied_move(ticker_symbol: str) -> float:
    try:
        tk = yf.Ticker(ticker_symbol)
        expirations = tk.options
        if not expirations:
            return 0.0
        chain = tk.option_chain(expirations[0])
        hist = tk.history(period="1d")
        if hist.empty:
            return 0.0
        current_price = hist["Close"].iloc[-1]
        
        atm_call = chain.calls.iloc[(chain.calls["strike"] - current_price).abs().argsort()[:1]]
        atm_put = chain.puts.iloc[(chain.puts["strike"] - current_price).abs().argsort()[:1]]
        
        straddle_cost = atm_call["lastPrice"].values[0] + atm_put["lastPrice"].values[0]
        return round((straddle_cost / current_price) * 100.0, 2)
    except Exception:
        return 0.0

@app.get("/api/health")
def health_check():
    return {"status": "ok", "environment": "vercel-serverless"}

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_earnings(req: PredictionRequest):
    ticker = req.ticker.upper()
    
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")

    try:
        # 1. Fetch Market Context & Implied Move
        implied_move = calculate_implied_move(ticker)
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        
        forward_eps = info.get("forwardEps", "N/A")
        recommendation = info.get("recommendationKey", "N/A")
        trailing_pe = info.get("trailingPE", "N/A")

        # 2. Query Gemini Structured Reasoning Model
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        prompt = f"""
        You are an institutional quantitative equity research agent specialized in post-earnings reactions.
        Analyze ticker {ticker} given the current metrics:
        - Front-Week Implied Options Move: {implied_move}%
        - Forward EPS Consensus: {forward_eps}
        - Analyst Consensus Stance: {recommendation}
        - Trailing P/E: {trailing_pe}
        
        Evaluate the 4 key drivers:
        1. Macro & Market Beta (Is overall liquidity expanding or contracting?)
        2. Surprise Delta (Priced-in whisper vs baseline consensus)
        3. Sector Guidance Trends (Supply chain lead times, customer capex shifts)
        4. Pre-Earnings Management Tone (Conservative posture vs confident signaling)

        Provide a calibrated forecast.
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PredictionResponse,
                temperature=0.2,
            ),
        )

        return response.parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
