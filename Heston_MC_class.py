import numpy as np
from typing import Tuple
from scipy.stats import norm
import warnings
#np.seterr(over='raise')
from scipy.interpolate import RegularGridInterpolator

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
            v_curr = np.clip(v[i-1],0,10)
            Z_S = Z[i-1,0]
            Z_v = Z[i-1,1]
            exp = (self.params.r-0.5*v_curr-self.params.q)*self.dt + np.sqrt(v_curr*self.dt)*Z_S
            S[i] = S[i-1] * np.exp(np.clip(exp,-500,500))
            v_euler = v_curr+ self.params.kappa * (self.params.theta - v_curr)*self.dt + self.params.sigma*np.sqrt(v_curr * self.dt)*Z_v
            v[i] = v_euler + self.params.sigma**2 / 4 * (Z_v**2-1)*self.dt
        
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
        if(std_error < 1e-10): 
            std_error = 0
        elif(std_error > 1):
            std_error = 1
        return payoffs, price, std_error
    
    # def build_analytical_interpolators(self, S_min, S_max, v_min, v_max, tau_max, K, K_U, grid_size):
    #     """
    #     Pre-computes analytical Greeks over a 3D grid (S, v, tau) and returns
    #     fast interpolator objects.
        
    #     Args:
    #         S_min, S_max: Bounds for the stock price
    #         v_min, v_max: Bounds for the variance
    #         tau_max: Maximum time to maturity (usually params.tau)
    #         grid_size: Number of points per dimension (25^3 = 15,625 integrations)
    #     """
    #     print(f"Building 3D Analytical Interpolation Grid ({grid_size}^3 points)...")
        
    #     # 1. Define strictly ascending grid points (required by RegularGridInterpolator)
    #     S_points = np.linspace(S_min, S_max, grid_size)
    #     v_points = np.linspace(max(1e-6, v_min), v_max, grid_size)
        
    #     # tau must be ascending, so we go from near-zero to tau_max
    #     tau_points = np.linspace(1e-6, tau_max, grid_size)

    #     # 2. Initialize 3D arrays to hold the pre-computed Greeks
    #     shape = (len(S_points), len(v_points), len(tau_points))
    #     delta_V_grid = np.zeros(shape)
    #     delta_U_grid = np.zeros(shape)
    #     vega_V_grid  = np.zeros(shape)
    #     vega_U_grid  = np.zeros(shape)

    #     # 3. Populate the grids (This runs ONCE before the simulation)
    #     for i, s in enumerate(S_points):
    #         for j, v in enumerate(v_points):
    #             for k, tau in enumerate(tau_points):
                    
    #                 # Call your heavy integration functions here
    #                 delta_V_grid[i, j, k] = analytic_heston_delta(S=s, vi=v, tau=tau, K=K,isP1=True)
    #                 delta_U_grid[i, j, k] = analytic_heston_delta(S=s, vi=v, tau=tau, K=K_U,isP1=True)
                    
    #                 vega_V_grid[i, j, k] = analytic_estimate_vega_fd(S=s, vi=v, tau=tau, K=K)
    #                 vega_U_grid[i, j, k] = analytic_estimate_vega_fd(S=s, vi=v, tau=tau, K=K_U)

    #     # 4. Create the Interpolator Objects
    #     # bounds_error=False and fill_value=None allows it to extrapolate 
    #     # safely if a Monte Carlo path jumps slightly outside the grid bounds.
    #     kwargs = {'bounds_error': False, 'fill_value': None}
        
    #     interp_delta_V = RegularGridInterpolator((S_points, v_points, tau_points), delta_V_grid, **kwargs)
    #     interp_delta_U = RegularGridInterpolator((S_points, v_points, tau_points), delta_U_grid, **kwargs)
    #     interp_vega_V  = RegularGridInterpolator((S_points, v_points, tau_points), vega_V_grid, **kwargs)
    #     interp_vega_U  = RegularGridInterpolator((S_points, v_points, tau_points), vega_U_grid, **kwargs)
        
    #     print("Interpolation Grid Built Successfully!")
    #     return interp_delta_V, interp_delta_U, interp_vega_V, interp_vega_U
    
    # def estimate_delta_finite_diff(self, M_simulations: int, N_paths_per_sim: int, tau_i: float, v_i: float, 
    #                           option_type: str = 'call', dS: float = 0.01, alpha: float = 0.05) -> Tuple[float, float]:
        
    #     S0_original = self.params.S0
    #     estimated_deltas = np.zeros(M_simulations)
    #     seed_history=[]
        
    #     # Run M independent simulations
    #     for m in range(M_simulations):
    #         seed = np.random.randint(0, 10e6)
    #         while(seed in seed_history):
    #             seed = np.random.randint(0, 10e6)
    #         seed_history.append(seed)
    #         # 2. Price at S0 + dS
    #         self.params = HestonParams(
    #             S0=S0_original + dS, K=self.params.K, K_U=self.params.K_U, r=self.params.r, q=self.params.q,
    #             v0=v_i, kappa=self.params.kappa, theta=self.params.theta,
    #             sigma=self.params.sigma, rho=self.params.rho, tau=tau_i, 
    #             analytical_delta=self.params.analytical_delta
    #         )
    #         _, price_up, _ = self.price_option(N_paths_per_sim, option_type, seed=seed)
            
    #         # 3. Price at S0 - dS
    #         self.params = HestonParams(
    #             S0=S0_original - dS, K=self.params.K, K_U=self.params.K_U, r=self.params.r, q=self.params.q,
    #             v0=v_i, kappa=self.params.kappa, theta=self.params.theta,
    #             sigma=self.params.sigma, rho=self.params.rho, tau=tau_i,
    #             analytical_delta=self.params.analytical_delta
    #         )
    #         _, price_down, _ = self.price_option(N_paths_per_sim, option_type, seed=seed)
            
    #         # Calculate and store the delta for this specific simulation
    #         estimated_deltas[m] = (price_up - price_down) / (2 * dS)
            
    #     # Restore original S0
    #     self.params = HestonParams(
    #         S0=S0_original, K=self.params.K, K_U=self.params.K_U, r=self.params.r, q=self.params.q,
    #         v0=v_i, kappa=self.params.kappa, theta=self.params.theta,
    #         sigma=self.params.sigma, rho=self.params.rho, tau=tau_i,
    #         analytical_delta=self.params.analytical_delta
    #     )
        
    #     # --- Statistics ---
        
    #     # 1. The final Delta estimate is the average of all macro-replications
    #     final_delta = np.mean(estimated_deltas)
        
    #     # 2. The standard error across the M independent simulations
    #     delta_std_error = np.std(estimated_deltas, ddof=1) / np.sqrt(M_simulations)

        
    #     return final_delta, delta_std_error
    
    # for one simulation only
    def estimate_greek_finite_diff(self, S_curr, N_paths_per_sim: int, tau_i: float, v_i: float, seed: int, 
                                   bump: float, greek: str, target: str = 'V',
                                   option_type: str = 'call') -> Tuple[float, float]:
        """
        Estimates Delta or Vega for either the target option (V) or hedging option (U)
        using central finite differences.
        """
        # Save original states
        S0_original = S_curr
        K_original = self.params.K
        v_original = self.params.v0
        
        # 1. Route the correct strike
        # By temporarily overriding 'K', your underlying price_option() method 
        # will naturally calculate the payoff for the correct option.
        active_K = self.params.K if target == 'V' else self.params.K_U
        
        # 2. Determine bumps based on which Greek we are calculating
        if greek == 'delta':
            S_up, S_down = S0_original + bump, S0_original - bump
            v_up, v_down = v_i, v_i
        elif greek == 'vega':
            S_up, S_down = S0_original, S0_original
            v_up = v_original + bump
            # Ensure variance doesn't bump into negative territory
            v_down = max(1e-6, v_original - bump) 
        else:
            raise ValueError("The 'greek' parameter must be either 'delta' or 'vega'.")

        # 3. Price the UP bump
        self.params = HestonParams(
            S0=S_up, K=active_K, K_U=self.params.K_U, r=self.params.r, q=self.params.q,
            v0=v_up, kappa=self.params.kappa, theta=self.params.theta,
            sigma=self.params.sigma, rho=self.params.rho, tau=tau_i, 
            analytical_delta=self.params.analytical_delta
        )
        _, price_up, std_up = self.price_option(N_paths_per_sim, option_type, seed)
        
        # 4. Price the DOWN bump
        self.params = HestonParams(
            S0=S_down, K=active_K, K_U=self.params.K_U, r=self.params.r, q=self.params.q,
            v0=v_down, kappa=self.params.kappa, theta=self.params.theta,
            sigma=self.params.sigma, rho=self.params.rho, tau=tau_i,
            analytical_delta=self.params.analytical_delta
        )
        _, price_down, std_down = self.price_option(N_paths_per_sim, option_type, seed)
        
        # 5. Restore original parameters
        self.params = HestonParams(
            S0=S0_original, K=K_original, K_U=self.params.K_U, r=self.params.r, q=self.params.q,
            v0=v_i, kappa=self.params.kappa, theta=self.params.theta,
            sigma=self.params.sigma, rho=self.params.rho, tau=tau_i,
            analytical_delta=self.params.analytical_delta
        )
        
        # 6. Calculate finite difference
        # We calculate the actual step taken in case v_down was floored at 1e-6
        actual_step = (S_up - S_down) if greek == 'delta' else (v_up - v_down)
        
        #print(price_up)
        #print(price_down)
        greek_val = (price_up - price_down) / actual_step
        #print(std_up)
        #print(std_down,"\n")
        greek_std = np.sqrt(std_up**2 + std_down**2) / actual_step 
        
        
        return greek_val, greek_std
    

    def estimate_delta_finite_diff(self, N_paths_per_sim: int, tau_i: float, v_i: float, seed, dS: float,
                               option_type: str = 'call') -> Tuple[float, float]:

        # Save original S0
        S0_original = self.params.S0
        
        # Price at S0 + dS
        self.params = HestonParams(
            S0=S0_original + dS, K=self.params.K, K_U = self.params.K_U, r=self.params.r, q=self.params.q,
            v0=v_i, kappa=self.params.kappa, theta=self.params.theta,
            sigma=self.params.sigma, rho=self.params.rho, tau=tau_i, 
            analytical_delta = self.params.analytical_delta
        )
        _, price_up, std_up = self.price_option(N_paths_per_sim, option_type, seed)
        
        # Price at S0 - dS
        self.params = HestonParams(
            S0=S0_original - dS, K=self.params.K, K_U = self.params.K_U, r=self.params.r, q=self.params.q,
            v0=v_i, kappa=self.params.kappa, theta=self.params.theta,
            sigma=self.params.sigma, rho=self.params.rho, tau=tau_i,
            analytical_delta = self.params.analytical_delta
        )
        _, price_down, std_down = self.price_option(N_paths_per_sim, option_type, seed)
        
        # Restore original S0
        self.params = HestonParams(
            S0=S0_original, K=self.params.K, K_U = self.params.K_U, r=self.params.r, q=self.params.q,
            v0=v_i, kappa=self.params.kappa, theta=self.params.theta,
            sigma=self.params.sigma, rho=self.params.rho, tau=tau_i,
            analytical_delta = self.params.analytical_delta
        )
        
        # Calculate delta
        delta = (price_up - price_down) / (2 * dS)
        
        # Approximate standard error
        delta_std = np.sqrt(std_up**2 + std_down**2) / (2 * dS)
        
        return delta, delta_std

    def get_bs_price(self, S, t_remaining, vol_proxy, strike = None) -> float:
        """Calculate Black-Scholes call option price."""
        K = self.params.K if strike is None else strike

        if t_remaining <= 1e-6:
            return max(S - K, 0)

        sigma = np.sqrt(self.params.v0) if vol_proxy is None else vol_proxy

        d1 = (np.log(S / K) +
            (self.params.r - self.params.q + 0.5 * sigma**2) * t_remaining) / \
            (sigma * np.sqrt(t_remaining))
        d2 = d1 - sigma * np.sqrt(t_remaining)

        return (S * np.exp(-self.params.q * t_remaining) * norm.cdf(d1) -
                K * np.exp(-self.params.r * t_remaining) * norm.cdf(d2))


    def get_bs_delta(self, S, t_remaining, vol_proxy=None, strike=None) -> float:
        """Calculate Black-Scholes delta."""
        K = self.params.K if strike is None else strike

        if t_remaining <= 1e-6:
            return 1.0 if S > K else 0.0

        sigma = np.sqrt(self.params.v0) if vol_proxy is None else vol_proxy
        d1 = (np.log(S / K) +
            (self.params.r - self.params.q + 0.5 * sigma**2) * t_remaining) / \
            (sigma * np.sqrt(t_remaining))

        return np.exp(-self.params.q * t_remaining) * norm.cdf(d1)


    def get_bs_vega(self, S, t_remaining, vol_proxy=None, strike=None) -> float:
        """Calculate Black-Scholes vega."""
        K = self.params.K if strike is None else strike

        if t_remaining <= 1e-6:
            return 0.0

        sigma = np.sqrt(self.params.v0) if vol_proxy is None else vol_proxy
        d1 = (np.log(S / K) +
            (self.params.r - self.params.q + 0.5 * sigma**2) * t_remaining) / \
            (sigma * np.sqrt(t_remaining))

        return S * np.exp(-self.params.q * t_remaining) * np.sqrt(t_remaining) * norm.pdf(d1)


    def get_bs_price_U(self, S, t_remaining, vol_proxy=None) -> float:
        """Black-Scholes price for hedging option U with strike K_U."""
        return self.get_bs_price(S, t_remaining, vol_proxy, strike=self.params.K_U)


    def get_bs_delta_U(self, S, t_remaining, vol_proxy=None) -> float:
        """Black-Scholes delta for hedging option U with strike K_U."""
        return self.get_bs_delta(S, t_remaining, vol_proxy, strike=self.params.K_U)


    def get_bs_vega_U(self, S, t_remaining, vol_proxy=None) -> float:
        """Black-Scholes vega for hedging option U with strike K_U."""
        return self.get_bs_vega(S, t_remaining, vol_proxy, strike=self.params.K_U)