import numpy as np
from typing import Tuple

from Heston_params_class import HestonParams

class HestonMonteCarlo:
    """
    Monte Carlo simulator for the Heston model using Euler discretization.
    """
    
    def __init__(self, params: HestonParams, N_steps: int = 100, 
                 variance_scheme: str = 'full_truncation'):
        """
        Initialize the simulator.
        
        Parameters:
        -----------
        params : HestonParams
            Model parameters
        N_steps : int
            Number of time steps
        variance_scheme : str
            How to handle negative variances:
            - 'full_truncation': max(v, 0)
            - 'reflection': |v|
        """
        self.params = params
        self.N_steps = N_steps
        self.dt = params.tau / N_steps
        self.variance_scheme = variance_scheme
        
    def simulate_path(self, seed: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate a single path of stock price and variance.
        
        Returns:
        --------
        S_path : ndarray
            Stock price path of length (N_steps + 1)
        v_path : ndarray
            Variance path of length (N_steps + 1)
        """
        if seed is not None:
            np.random.seed(seed)
        
         # Time grid
        t_grid = np.linspace(0, self.params.tau, self.N_steps + 1)

        mu = np.array([0,0])
        cov = np.array([[1,self.params.rho],[self.params.rho,1]])

        S = np.full(shape=(self.N_steps+1), fill_value = self.params.S0)
        v = np.full(shape=(self.N_steps+1), fill_value = self.params.v0)
        Z = np.random.multivariate_normal(mu,cov,self.N_steps)
        for i in range(1,self.N_steps+1):
            # Milstein Scheme
            # https://dms.umontreal.ca/~bedard/Heston.pdf
            v_curr = max(v[i-1],0)
            Z_S = Z[i-1,0]
            Z_v = Z[i-1,1]
            S[i] = S[i-1] * np.exp((self.params.r-0.5*v_curr)*self.dt + np.sqrt(v_curr*self.dt)*Z_S)
            v_euler = v_curr+ self.params.kappa * (self.params.theta - v_curr)*self.dt + self.params.sigma*np.sqrt(v_curr * self.dt)*Z_v
            v[i] = np.maximum(v_euler + self.params.sigma**2 / 4 * (Z_v-1)*self.dt, 0)         
        
        return t_grid, S, v
    
    def price_option(self, N_paths: int, option_type: str = 'call', 
                    seed: int = None) -> Tuple[float, float]:
        if seed is not None:
            np.random.seed(seed)
        
        payoffs = np.zeros(N_paths)
        
        for i in range(N_paths):
            t_grid, S_path, _ = self.simulate_path()
            S_T = S_path[-1]
            
            if option_type == 'call':
                payoffs[i] = max(0, S_T - self.params.K)
            elif option_type == 'put':
                payoffs[i] = max(0, self.params.K - S_T)
            else:
                raise ValueError(f"Unknown option type: {option_type}")
        
        # Discount and take expectation
        df = np.exp(-self.params.r * self.params.tau)
        price = df * np.mean(payoffs)
        std_error = df * np.std(payoffs) / np.sqrt(N_paths)
        
        return payoffs, price, std_error
    
    def estimate_delta_finite_diff(self, N_paths: int, option_type: str = 'call',
                                  dS: float = 0.01, seed: int = None) -> Tuple[float, float]:

        # Save original S0
        S0_original = self.params.S0
        
        # Price at S0 + dS
        self.params = HestonParams(
            S0=S0_original + dS, K=self.params.K, r=self.params.r, q=self.params.q,
            v0=self.params.v0, kappa=self.params.kappa, theta=self.params.theta,
            sigma=self.params.sigma, rho=self.params.rho, tau=self.params.tau, 
            analytical_delta = self.params.analytical_delta
        )
        _, price_up, std_up = self.price_option(N_paths, option_type, seed)
        
        # Price at S0 - dS
        self.params = HestonParams(
            S0=S0_original - dS, K=self.params.K, r=self.params.r, q=self.params.q,
            v0=self.params.v0, kappa=self.params.kappa, theta=self.params.theta,
            sigma=self.params.sigma, rho=self.params.rho, tau=self.params.tau,
            analytical_delta = self.params.analytical_delta
        )
        _, price_down, std_down = self.price_option(N_paths, option_type, seed)
        
        # Restore original S0
        self.params = HestonParams(
            S0=S0_original, K=self.params.K, r=self.params.r, q=self.params.q,
            v0=self.params.v0, kappa=self.params.kappa, theta=self.params.theta,
            sigma=self.params.sigma, rho=self.params.rho, tau=self.params.tau,
            analytical_delta = self.params.analytical_delta
        )
        
        # Calculate delta
        delta = (price_up - price_down) / (2 * dS)
        
        # Approximate standard error
        delta_std = np.sqrt(std_up**2 + std_down**2) / (2 * dS)
        
        return delta, delta_std

    def get_bs_delta(self, S, t_remaining, vol_proxy=None)-> float:
        """
        Calculate Black-Scholes delta. 
        vol_proxy: defaults to sqrt(v0) (initial volatility) if None
        """
        from scipy.stats import norm
        
        if t_remaining <= 1e-6:
            return 1.0 if S > self.params.K else 0.0
            
        sigma = np.sqrt(self.params.v0) if vol_proxy is None else vol_proxy
        d1 = (np.log(S / self.params.K) + 
            (self.params.r - self.params.q + 0.5 * sigma**2) * t_remaining) / (sigma * np.sqrt(t_remaining))
        
        return np.exp(-self.params.q * t_remaining) * norm.cdf(d1)
    
    def get_bs_price(self, S, t_remaining, vol_proxy) -> float:
        """Calculate Black-Scholes call option price."""
        from scipy.stats import norm
        
        if t_remaining <= 1e-6:
            return max(S - self.params.K, 0)
        
        sigma = np.sqrt(self.params.v0) if vol_proxy is None else vol_proxy
        
        d1 = (np.log(S / self.params.K) + 
            (self.params.r - self.params.q + 0.5 * sigma**2) * t_remaining) / \
            (sigma * np.sqrt(t_remaining))
        d2 = d1 - sigma * np.sqrt(t_remaining)
        
        return (S * np.exp(-self.params.q * t_remaining) * norm.cdf(d1) - 
                self.params.K * np.exp(-self.params.r * t_remaining) * norm.cdf(d2))