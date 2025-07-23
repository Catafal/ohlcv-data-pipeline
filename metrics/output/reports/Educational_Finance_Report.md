# 📊 Complete Financial Analysis Report: The Magnificent 7 Stocks
*An Educational Guide to Understanding Investment Risk and Performance*

---

## 🎯 Executive Summary

This report analyzes 10 years of stock market data (2015-2025) for the "Magnificent 7" technology companies: **Apple (AAPL)**, **Amazon (AMZN)**, **Google (GOOGL)**, **Meta (META)**, **Microsoft (MSFT)**, **NVIDIA (NVDA)**, and **Tesla (TSLA)**. 

I examined **17,584 data points** to understand how these investments performed during both good times and market crashes, providing insights for new investors about risk, return, and portfolio diversification.

---

## 📈 What I Analyzed and Why

### 🔍 **The Companies**
- **AAPL (Apple)**: Consumer electronics, iPhones, iPads
- **AMZN (Amazon)**: E-commerce, cloud computing (AWS)
- **GOOGL (Google)**: Search, advertising, cloud services
- **META (Meta/Facebook)**: Social media, virtual reality
- **MSFT (Microsoft)**: Software, cloud computing, productivity tools
- **NVDA (NVIDIA)**: Graphics cards, AI chips, gaming
- **TSLA (Tesla)**: Electric vehicles, energy storage

### 📊 **Key Metrics I Calculated**

#### **1. Volatility (How Much Prices Jump Around)**
*Think of this as "bumpiness" - higher numbers mean more dramatic price swings*

| Company | Volatility | What This Means |
|---------|------------|-----------------|
| MSFT | 27.2% | 🟢 Most stable - prices change gradually |
| AAPL | 37.6% | 🟡 Moderate volatility |
| META | 38.4% | 🟡 Moderate volatility |
| GOOGL | 41.6% | 🟠 Getting bumpy |
| AMZN | 44.5% | 🟠 Quite volatile |
| NVDA | 62.5% | 🔴 Very volatile - big swings |
| TSLA | 67.5% | 🔴 Extremely volatile - roller coaster! |

**Portfolio (all 7 combined)**: 31.0% - Diversification reduces risk!

#### **2. Sharpe Ratio (Risk-Adjusted Returns)**
*This tells you how much extra return you got for taking extra risk. Higher is better.*

| Company | Sharpe Ratio | Rating |
|---------|--------------|--------|
| MSFT | 0.95 | 🏆 Excellent - great returns for the risk |
| NVDA | 0.84 | 🥇 Very good |
| Portfolio | 0.80 | 🥇 Very good (better than most individual stocks!) |
| META | 0.67 | 🥈 Good |
| TSLA | 0.44 | 🥉 Okay |
| AAPL | 0.37 | 🥉 Okay |
| AMZN | 0.34 | ⚠️ Below average |
| GOOGL | 0.25 | ⚠️ Below average |

#### **3. Maximum Drawdown (Worst Loss from Peak)**
*This shows the biggest loss you would have experienced if you bought at the worst possible time*

| Company | Max Drawdown | Real-World Meaning |
|---------|--------------|-------------------|
| MSFT | 37.6% | 🟢 If you invested $1000, worst case you'd be down to $624 |
| META | 76.7% | 🟠 Worst case: $1000 → $233 |
| AAPL | 78.9% | 🔴 Worst case: $1000 → $211 |
| NVDA | 92.3% | 🔴 Worst case: $1000 → $77 |
| TSLA | 95.2% | 🔴 Worst case: $1000 → $48 |
| AMZN | 97.8% | 🔴 Worst case: $1000 → $22 |
| GOOGL | 97.2% | 🔴 Worst case: $1000 → $28 |

**Portfolio**: 65.7% - Much better than individual stocks!

#### **4. Value at Risk (VaR) - Daily Loss Probability**
*On a typical bad day (happens 5% of the time), how much could you lose?*

| Company | 95% VaR | What This Means |
|---------|---------|-----------------|
| MSFT | 2.6% | On a bad day, might lose 2.6% |
| GOOGL | 2.8% | On a bad day, might lose 2.8% |
| AAPL | 2.8% | On a bad day, might lose 2.8% |
| Portfolio | 3.1% | Diversified portfolio risk |
| AMZN | 3.2% | On a bad day, might lose 3.2% |
| META | 3.6% | On a bad day, might lose 3.6% |
| NVDA | 4.6% | On a bad day, might lose 4.6% |
| TSLA | 5.5% | On a bad day, might lose 5.5% |

---

## 🌊 Historical Market Events Analysis

I studied how these stocks performed during 6 major market events:

### 1. 🦠 **COVID-19 Pandemic Crash (Feb-Mar 2020)**
- **Duration**: 33 days of pure terror
- **What Happened**: Global lockdowns, economic uncertainty
- **Winners**: Amazon (-12%) - people shopped online more
- **Losers**: Tesla (-53%) - luxury purchases delayed
- **Portfolio Loss**: -31% (recovered within months)

### 2. 📉 **Tech Selloff & Growth Bear Market (Nov 2021 - Oct 2022)**
- **Duration**: 338 days of sustained decline
- **What Happened**: Interest rates rose, tech valuations crashed
- **Key Insight**: Amazon and Google lost 97% of gains - nearly back to starting prices!
- **Portfolio Loss**: -59% (diversification helped vs individual stocks)

### 3. 🎢 **December 2018 Market Correction**
- **Duration**: 82 days of volatility
- **What Happened**: Trade war fears, economic uncertainty
- **Surprising Winner**: Tesla (+0.2%) - only stock that gained!
- **Biggest Loser**: NVIDIA (-56%) - crypto crash affected GPU demand

### 4. ⚡ **August 2015 Flash Crash**
- **Duration**: 7 days of rapid decline
- **What Happened**: China currency devaluation fears
- **Impact**: All stocks lost 10-15% quickly, but recovered fast

### 5. 📱 **May 2022 Growth Stock Crash**
- **Duration**: 72 days of tech stock punishment
- **What Happened**: Interest rate fears, growth stock rotation
- **Biggest Loser**: Amazon (-97%) - growth stocks hit hardest

### 6. 🚀 **Post-COVID Recovery (Mar-Aug 2020)**
- **Duration**: 148 days of explosive growth
- **What Happened**: Stimulus money, digitalization acceleration
- **Big Winner**: Tesla (+335%) - EV revolution began
- **Portfolio Gain**: +108% in less than 5 months!

---

## 📊 Visual Analysis Insights

### 🔥 **Correlation Heatmap Analysis**
My correlation analysis reveals important diversification insights:

- **Tesla (TSLA)**: Lowest correlations (0.23-0.37) - moves independently
- **Microsoft (MSFT)**: Higher correlations (0.49-0.60) - moves with market
- **Best Diversifier**: Tesla, despite high volatility
- **Most Correlated**: MSFT and other established tech stocks

*Although we have to take into account that those are EEUU companies, so it will have correlation in one way or another between them.*

### 📉 **Drawdown Analysis Insights**
The drawdown chart clearly shows:

- **2020 COVID Crash**: Sharp, deep, but quick recovery
- **2021-2022 Tech Selloff**: Prolonged, grinding decline
- **Recovery Patterns**: Tech stocks can recover dramatically
- **Portfolio Protection**: Diversification reduces maximum losses

### 🎲 **Monte Carlo Simulation Results**
My 1,000 scenario simulation tested three market conditions:

1. **Normal Markets**: Expected moderate volatility
2. **Stressed Markets**: Higher volatility, lower returns
3. **High Volatility**: Extreme market conditions

**Key Finding**: Portfolio performs better than individual stocks in ALL scenarios.

---

## 🎯 Investment Recommendations for Beginners

### 🟢 **Conservative Approach**
- **Primary Holdings**: Microsoft (40%), Apple (30%)
- **Why**: Lower volatility, consistent returns
- **Expected**: Moderate returns with manageable risk

### 🟡 **Balanced Approach**
- **Equal Weight Portfolio**: All 7 stocks equally weighted
- **Why**: Balances growth potential with diversification
- **Expected**: Good returns with moderate risk

### 🔴 **Aggressive Approach**
- **Growth Focus**: NVIDIA (30%), Tesla (25%), Meta (25%), Others (20%)
- **Why**: Higher growth potential
- **Warning**: Much higher volatility and potential losses

---


## 📚 Glossary

- **Volatility**: How much a stock price bounces up and down
- **Sharpe Ratio**: Return per unit of risk (higher = better)
- **Drawdown**: Loss from the highest point to lowest point
- **VaR (Value at Risk)**: Potential loss on a bad day
- **Correlation**: How much stocks move together (1.0 = perfectly together, 0 = independent)
- **Portfolio**: Collection of different investments
- **Diversification**: Spreading risk across multiple investments

---

**Disclaimer**: This analysis is for educational purposes only. Past performance does not guarantee future results. Always consult with a financial advisor before making investment decisions.