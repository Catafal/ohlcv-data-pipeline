"""
Scenario Definitions

Defines historical market scenarios and crisis periods for stress testing analysis.
Each scenario includes specific date ranges, market characteristics, and expected impacts.

This module contains predefined scenarios based on major historical market events,
allowing for systematic analysis of how securities perform during various crisis types.

Scenarios include:
- COVID-19 Pandemic Crash (2020)
- Tech Stock Selloff (2021-2022)  
- December 2018 Correction
- August 2015 Flash Crash
- May 2022 Growth Stock Crisis
- Custom user-defined scenarios

Author: OHLCV Data Pipeline - Scenario Analysis Module
"""

from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from enum import Enum

class ScenarioType(Enum):
    """Enumeration of different types of market scenarios."""
    CRASH = "crash"                    # Rapid, severe market declines
    CORRECTION = "correction"          # Moderate market pullbacks (10-20%)
    BEAR_MARKET = "bear_market"       # Extended declining markets (>20%, >2 months)
    VOLATILITY_SPIKE = "volatility_spike"  # Sharp increases in market volatility
    SECTOR_ROTATION = "sector_rotation"    # Specific sector underperformance
    FLASH_CRASH = "flash_crash"           # Very rapid intraday crashes
    RECOVERY = "recovery"                 # Post-crisis recovery periods

@dataclass
class MarketScenario:
    """
    Represents a specific market scenario for stress testing.
    
    Contains all necessary information to analyze how securities performed
    during a particular historical period or simulated stress condition.
    """
    name: str                           # Human-readable scenario name
    scenario_id: str                    # Unique identifier for the scenario
    scenario_type: ScenarioType         # Type of market scenario
    start_date: date                    # Scenario start date
    end_date: date                      # Scenario end date
    description: str                    # Detailed description of the scenario
    key_characteristics: List[str]      # List of key market characteristics
    expected_impact: Dict[str, str]     # Expected impact on different asset classes
    severity_level: int                 # Severity scale 1-10 (10 = most severe)
    market_backdrop: str                # Overall market conditions during period
    primary_drivers: List[str]          # Main factors driving the scenario
    
    @property
    def duration_days(self) -> int:
        """Calculate scenario duration in days."""
        return (self.end_date - self.start_date).days
    
    @property
    def year(self) -> int:
        """Get the primary year of the scenario."""
        return self.start_date.year

class ScenarioLibrary:
    """
    Library of predefined historical market scenarios for stress testing.
    
    This class provides a comprehensive collection of historically significant
    market periods that can be used for systematic scenario analysis.
    """
    
    def __init__(self):
        """Initialize the scenario library with predefined scenarios."""
        self._scenarios = self._initialize_scenarios()
    
    def _initialize_scenarios(self) -> Dict[str, MarketScenario]:
        """
        Initialize the library with predefined historical scenarios.
        
        Returns:
            Dictionary mapping scenario IDs to MarketScenario objects
        """
        scenarios = {}
        
        # COVID-19 Pandemic Crash (February-March 2020)
        scenarios['covid_crash_2020'] = MarketScenario(
            name="COVID-19 Pandemic Crash",
            scenario_id="covid_crash_2020",
            scenario_type=ScenarioType.CRASH,
            start_date=date(2020, 2, 19),  # Market peak before crash
            end_date=date(2020, 3, 23),    # Market trough
            description="Rapid 35% market decline in 33 days due to COVID-19 pandemic lockdowns and economic uncertainty",
            key_characteristics=[
                "Fastest bear market in history",
                "Global lockdowns and travel restrictions", 
                "Massive fiscal and monetary stimulus response",
                "Technology stocks initially fell but recovered quickly",
                "Energy and travel stocks hit hardest"
            ],
            expected_impact={
                "large_cap_tech": "Initially severe decline, then rapid recovery",
                "travel_hospitality": "Severe and prolonged decline",
                "healthcare": "Mixed performance, vaccine companies outperformed",
                "energy": "Severe decline due to demand destruction",
                "growth_stocks": "V-shaped recovery after initial decline"
            },
            severity_level=9,
            market_backdrop="Global pandemic causing unprecedented economic shutdown",
            primary_drivers=[
                "COVID-19 pandemic spread",
                "Economic lockdowns",
                "Supply chain disruptions", 
                "Flight to safety",
                "Massive monetary stimulus"
            ]
        )
        
        # 2021-2022 Tech Selloff / Growth Stock Bear Market
        scenarios['tech_selloff_2021_2022'] = MarketScenario(
            name="Tech Selloff & Growth Stock Bear Market",
            scenario_id="tech_selloff_2021_2022", 
            scenario_type=ScenarioType.BEAR_MARKET,
            start_date=date(2021, 11, 8),   # Peak of many growth stocks
            end_date=date(2022, 10, 12),    # Approximate trough
            description="Extended bear market in growth and technology stocks due to rising interest rates and inflation concerns",
            key_characteristics=[
                "Rising interest rates and inflation",
                "Multiple compression in growth stocks",
                "Rotation from growth to value stocks",
                "Cryptocurrency crash coincided",
                "Supply chain and geopolitical tensions"
            ],
            expected_impact={
                "growth_stocks": "Severe bear market with 50-80% declines",
                "mega_cap_tech": "Significant but less severe declines", 
                "value_stocks": "Relative outperformance",
                "energy": "Strong outperformance due to inflation",
                "utilities": "Defensive characteristics provided some protection"
            },
            severity_level=7,
            market_backdrop="Rising rates and inflation ending easy money era",
            primary_drivers=[
                "Federal Reserve rate hikes",
                "Inflation concerns",
                "Multiple compression",
                "Geopolitical tensions (Russia-Ukraine)",
                "Supply chain disruptions"
            ]
        )
        
        # December 2018 Market Correction
        scenarios['december_2018_correction'] = MarketScenario(
            name="December 2018 Market Correction",
            scenario_id="december_2018_correction",
            scenario_type=ScenarioType.CORRECTION,
            start_date=date(2018, 10, 3),   # Market peak
            end_date=date(2018, 12, 24),    # Christmas Eve low
            description="Sharp market correction driven by Fed rate hikes, trade war fears, and economic growth concerns",
            key_characteristics=[
                "Federal Reserve tightening cycle",
                "Trade war tensions with China",
                "Yield curve flattening concerns",
                "Q4 earnings weakness",
                "Oil price collapse"
            ],
            expected_impact={
                "all_sectors": "Broad-based selling pressure",
                "technology": "Significant decline amid growth concerns",
                "financials": "Pressured by yield curve flattening",
                "energy": "Additional pressure from oil price collapse",
                "consumer_discretionary": "Weakness on economic growth fears"
            },
            severity_level=6,
            market_backdrop="Late-cycle economic concerns and monetary tightening",
            primary_drivers=[
                "Federal Reserve rate hikes",
                "US-China trade tensions",
                "Economic growth slowdown fears",
                "Yield curve inversion concerns",
                "Oil price volatility"
            ]
        )
        
        # August 2015 Flash Crash
        scenarios['august_2015_flash_crash'] = MarketScenario(
            name="August 2015 Flash Crash",
            scenario_id="august_2015_flash_crash",
            scenario_type=ScenarioType.FLASH_CRASH,
            start_date=date(2015, 8, 18),   # Week before crash
            end_date=date(2015, 8, 25),     # Recovery began
            description="Sharp market decline triggered by China yuan devaluation and concerns about Chinese economic growth",
            key_characteristics=[
                "China yuan devaluation shock",
                "Emerging market currency crisis",
                "Commodity price collapse",
                "High-frequency trading amplified moves",
                "Flash crash characteristics in individual stocks"
            ],
            expected_impact={
                "emerging_markets": "Severe decline on currency concerns",
                "commodities": "Sharp decline on China demand fears",
                "multinational_companies": "Currency headwinds",
                "technology": "Moderate decline but quick recovery",
                "defensive_sectors": "Relative outperformance"
            },
            severity_level=6,
            market_backdrop="Chinese economic slowdown and currency devaluation",
            primary_drivers=[
                "China yuan devaluation",
                "Chinese economic growth concerns",
                "Commodity price weakness",
                "Emerging market contagion",
                "High-frequency trading volatility"
            ]
        )
        
        # May 2022 Growth Stock Crash
        scenarios['may_2022_growth_crash'] = MarketScenario(
            name="May 2022 Growth Stock Crash",
            scenario_id="may_2022_growth_crash",
            scenario_type=ScenarioType.CRASH,
            start_date=date(2022, 4, 5),    # Pre-crash levels
            end_date=date(2022, 6, 16),     # Approximate trough
            description="Severe decline in growth stocks amid aggressive Fed tightening and inflation fears",
            key_characteristics=[
                "Aggressive Federal Reserve tightening",
                "Inflation at 40-year highs",
                "Growth stock multiple compression",
                "Cryptocurrency crash coincided",
                "Margin calls and forced selling"
            ],
            expected_impact={
                "growth_stocks": "Severe 30-50% declines",
                "technology": "Broad-based significant decline",
                "crypto_exposed": "Extreme volatility and declines",
                "unprofitable_companies": "Especially severe declines",
                "defensive_stocks": "Relative outperformance"
            },
            severity_level=8,
            market_backdrop="Inflation surge forcing aggressive monetary tightening",
            primary_drivers=[
                "Federal Reserve hawkishness",
                "Inflation surge to 9%+",
                "Multiple compression",
                "Liquidity concerns",
                "Economic recession fears"
            ]
        )
        
        # Post-COVID Recovery (for comparison)
        scenarios['covid_recovery_2020'] = MarketScenario(
            name="Post-COVID Market Recovery",
            scenario_id="covid_recovery_2020",
            scenario_type=ScenarioType.RECOVERY,
            start_date=date(2020, 3, 23),   # Market trough
            end_date=date(2020, 8, 18),     # New highs achieved
            description="Rapid V-shaped recovery driven by unprecedented fiscal and monetary stimulus",
            key_characteristics=[
                "Massive fiscal stimulus packages",
                "Federal Reserve emergency measures",
                "Technology adoption acceleration",
                "Stay-at-home stock outperformance",
                "SPAC and meme stock mania began"
            ],
            expected_impact={
                "technology": "Massive outperformance",
                "stay_at_home": "Exceptional gains",
                "travel_leisure": "Lagged recovery",
                "small_caps": "Strong performance", 
                "growth_stocks": "Significant outperformance vs value"
            },
            severity_level=2,  # Low severity as it's a recovery
            market_backdrop="Unprecedented stimulus driving rapid recovery",
            primary_drivers=[
                "Fiscal stimulus packages",
                "Federal Reserve liquidity",
                "Vaccine development optimism",
                "Technology adoption",
                "Reopening expectations"
            ]
        )
        
        return scenarios
    
    def get_scenario(self, scenario_id: str) -> Optional[MarketScenario]:
        """
        Get a specific scenario by ID.
        
        Args:
            scenario_id: Unique identifier for the scenario
            
        Returns:
            MarketScenario object or None if not found
        """
        return self._scenarios.get(scenario_id)
    
    def get_all_scenarios(self) -> Dict[str, MarketScenario]:
        """
        Get all available scenarios.
        
        Returns:
            Dictionary of all scenarios
        """
        return self._scenarios.copy()
    
    def get_scenarios_by_type(self, scenario_type: ScenarioType) -> List[MarketScenario]:
        """
        Get all scenarios of a specific type.
        
        Args:
            scenario_type: Type of scenarios to retrieve
            
        Returns:
            List of matching scenarios
        """
        return [scenario for scenario in self._scenarios.values() 
                if scenario.scenario_type == scenario_type]
    
    def get_scenarios_by_year(self, year: int) -> List[MarketScenario]:
        """
        Get all scenarios that occurred in a specific year.
        
        Args:
            year: Year to filter by
            
        Returns:
            List of scenarios from that year
        """
        return [scenario for scenario in self._scenarios.values() 
                if scenario.year == year]
    
    def get_scenarios_by_severity(self, min_severity: int = 1, max_severity: int = 10) -> List[MarketScenario]:
        """
        Get scenarios within a specific severity range.
        
        Args:
            min_severity: Minimum severity level (1-10)
            max_severity: Maximum severity level (1-10)
            
        Returns:
            List of scenarios within severity range
        """
        return [scenario for scenario in self._scenarios.values() 
                if min_severity <= scenario.severity_level <= max_severity]
    
    def create_custom_scenario(self,
                             name: str,
                             scenario_id: str,
                             start_date: date,
                             end_date: date,
                             scenario_type: ScenarioType = ScenarioType.CORRECTION,
                             description: str = "",
                             severity_level: int = 5) -> MarketScenario:
        """
        Create a custom scenario for analysis.
        
        Args:
            name: Human-readable name for the scenario
            scenario_id: Unique identifier
            start_date: Scenario start date
            end_date: Scenario end date
            scenario_type: Type of scenario
            description: Detailed description
            severity_level: Severity rating 1-10
            
        Returns:
            New MarketScenario object
        """
        custom_scenario = MarketScenario(
            name=name,
            scenario_id=scenario_id,
            scenario_type=scenario_type,
            start_date=start_date,
            end_date=end_date,
            description=description or f"Custom scenario: {name}",
            key_characteristics=["Custom scenario"],
            expected_impact={"all_assets": "To be determined"},
            severity_level=severity_level,
            market_backdrop="User-defined period",
            primary_drivers=["User-defined factors"]
        )
        
        # Add to scenarios library
        self._scenarios[scenario_id] = custom_scenario
        
        return custom_scenario
    
    def get_scenario_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about available scenarios.
        
        Returns:
            Dictionary with scenario library statistics
        """
        scenarios = list(self._scenarios.values())
        
        if not scenarios:
            return {}
        
        # Count by type
        type_counts = {}
        for scenario_type in ScenarioType:
            count = len([s for s in scenarios if s.scenario_type == scenario_type])
            if count > 0:
                type_counts[scenario_type.value] = count
        
        # Severity distribution
        severity_levels = [s.severity_level for s in scenarios]
        
        # Date range
        start_dates = [s.start_date for s in scenarios]
        end_dates = [s.end_date for s in scenarios]
        
        summary = {
            'total_scenarios': len(scenarios),
            'scenario_types': type_counts,
            'severity_range': {'min': min(severity_levels), 'max': max(severity_levels)},
            'date_range': {
                'earliest_start': min(start_dates).strftime('%Y-%m-%d'),
                'latest_end': max(end_dates).strftime('%Y-%m-%d')
            },
            'scenario_list': [{'id': s.scenario_id, 'name': s.name, 'type': s.scenario_type.value, 
                             'severity': s.severity_level} for s in scenarios]
        }
        
        return summary