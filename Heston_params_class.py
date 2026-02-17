from dataclasses import dataclass
@dataclass
class HestonParams:
    """Heston model parameters"""
    S0: float      # Initial stock price
    K: float       # Strike price
    K_U: float     # Strike for hedging option U
    r: float       # Risk-free rate
    q: float       # Dividend yield
    v0: float      # Initial variance
    kappa: float   # Mean reversion speed
    theta: float   # Long-term variance
    sigma: float   # Volatility of variance (vol of vol)
    rho: float     # Correlation
    tau: float     # Time to maturity
    analytical_delta: float
    lambd: float = 0.0
