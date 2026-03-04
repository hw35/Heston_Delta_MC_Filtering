import numpy as np
from dataclasses import dataclass
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
@dataclass
class HestonHedging:
    def simulate_delta_hedging(self,simulator, seed, rehedge_steps=1):
        """
        Simulates a delta-hedged path for a Short Call position.
        
        Args:
            rehedge_steps (int): Number of time steps between rebalancing (1 = every step)
        """
        # 1. Simulate the Real World Path (Heston dynamics)
        #t_grid = np.linspace(0, simulator.params.tau, simulator.N_steps + 1)
        t_grid, S, v = simulator.simulate_path(seed=seed)
        
        # 2. Initialize Portfolio
        # Position: Short 1 Call, Long Delta Shares, Cash Account
        portfolio_value = np.zeros(len(t_grid))
        deltas = np.zeros(len(t_grid))
        stock_holdings = np.zeros(len(t_grid))
        cash = np.zeros(len(t_grid))
        
        # Initial Setup (t=0)
        # We assume we sell the option at the BS theoretical price (or Heston price if preferred)
        # For P&L tracking relative to hedge, we usually start with 0 P&L or strictly track cost.
        # Here we track the "Hedge Portfolio" vs "Option Liability"
        
        tau = simulator.params.tau
        deltas[0] = simulator.get_bs_delta(S[0], tau)
        stock_holdings[0] = deltas[0]
        
        # FIXED: Account for option premium received when selling the call
        initial_option_price = simulator.get_bs_price(S[0], tau, None)
        cash[0] = initial_option_price - (stock_holdings[0] * S[0]) 
        
        # 3. Step through time
        for i in range(1, len(t_grid)):
            dt = t_grid[i] - t_grid[i-1]
            tau = simulator.params.tau - t_grid[i]
            
            # Accrue interest on cash
            cash[i] = cash[i-1] * np.exp(simulator.params.r * dt)
            
            # Calculate new Delta
            current_delta = simulator.get_bs_delta(S[i], tau)
            deltas[i] = current_delta
            
            # Rehedge logic
            if i % rehedge_steps == 0 and i < len(t_grid) - 1:
                # Buy/Sell stock to match new delta
                shares_to_buy = current_delta - stock_holdings[i-1]
                cost = shares_to_buy * S[i]
                cash[i] -= cost
                stock_holdings[i] = current_delta
            else:
                # Carry over previous position
                stock_holdings[i] = stock_holdings[i-1]
                
            # Calculate Portfolio Value (Stock + Cash) vs Option Liability
            # Option Liability (Intrinsic value at maturity, approx BS value before)
            # For trajectory visualization, simple intrinsic is often enough at T, 
            # but using BS price during path gives smoother P&L.
            
            # Note: A perfect hedge would result in Portfolio Value == Option Price
            portfolio_value[i] = stock_holdings[i] * S[i] + cash[i]

        return t_grid, S, v, deltas, portfolio_value

    def delta_vega_hedging(self, simulator, seed, rehedge_steps):
        """
        Simulates a delta-vega-hedged path for a Short Call position.
        Implements the full Heston hedging portfolio: Pi = V + Delta*S + phi*U
        
        The portfolio holds:
            - Short 1 target option V (e.g. ATM call)
            - Delta shares of stock S (to hedge dS risk)
            - phi units of hedging option U (to hedge dv risk)
            - Cash account
        
        Args:
            rehedge_steps (int): Number of time steps between rebalancing
        """
        # 1. Simulate the Real World Path (Heston dynamics)
        t_grid, S, v = simulator.simulate_path(seed=seed)

        # 2. Initialize arrays
        portfolio_value  = np.zeros(len(t_grid))
        deltas           = np.zeros(len(t_grid))
        vegas_V          = np.zeros(len(t_grid))  # vega of target option V
        vegas_U          = np.zeros(len(t_grid))  # vega of hedging option U
        phis             = np.zeros(len(t_grid))  # units of hedging option U
        stock_holdings   = np.zeros(len(t_grid))
        option_U_holdings= np.zeros(len(t_grid))
        cash             = np.zeros(len(t_grid))

        # 3. Initial Setup (t=0)
        tau = simulator.params.tau

        # Price and Greeks of target option V
        V0 = simulator.get_bs_price(S[0], tau, v[0])
        deltas[0] = simulator.get_bs_delta(S[0], tau)
        vegas_V[0] = simulator.get_bs_vega(S[0], tau)

        # Price and Greeks of hedging option U (different strike/maturity)
        U0 = simulator.get_bs_price_U(S[0], tau, v[0])
        vegas_U[0] = simulator.get_bs_vega_U(S[0], tau)

        # phi: chosen so that phi * vega_U cancels vega_V
        # From the Heston PDE hedge condition: vega_V + phi * vega_U = 0
        phis[0] = -vegas_V[0] / vegas_U[0]
        stock_holdings[0]  = deltas[0]
        option_U_holdings[0] = phis[0]

        # Self-financing: cash = premiums received minus cost of hedge instruments
        # Short V (receive V0), long phi units of U (cost phi*U0), long delta shares (cost delta*S0)
        cash[0] = V0 - (stock_holdings[0] * S[0]) - (option_U_holdings[0] * U0)

        # 4. Step through time
        for i in range(1, len(t_grid)):
            dt  = t_grid[i] - t_grid[i-1]
            tau = simulator.params.tau - t_grid[i]

            # Accrue interest on cash
            cash[i] = cash[i-1] * np.exp(simulator.params.r * dt)

            # Current prices and Greeks
            V_i = simulator.get_bs_price(S[i], tau, v[i])
            U_i = simulator.get_bs_price_U(S[i], tau, v[i])

            current_delta   = simulator.get_bs_delta(S[i], tau)
            current_vega_V  = simulator.get_bs_vega(S[i], tau)
            current_vega_U  = simulator.get_bs_vega_U(S[i], tau)
            if (abs(current_vega_U) == 0 or abs(-current_vega_V / current_vega_U) > 100):
                current_phi = phis[i-1]
            else:
                current_phi = -current_vega_V / current_vega_U

            deltas[i]   = current_delta
            vegas_V[i]  = current_vega_V
            vegas_U[i]  = current_vega_U
            phis[i]     = current_phi

            # Rehedge logic
            if i % rehedge_steps == 0 and i < len(t_grid) - 1:
                # Rebalance stock position
                shares_to_trade = current_delta - stock_holdings[i-1]
                cash[i] -= shares_to_trade * S[i]
                stock_holdings[i] = current_delta

                # Rebalance hedging option U position
                options_to_trade = current_phi - option_U_holdings[i-1]
                cash[i] -= options_to_trade * U_i
                option_U_holdings[i] = current_phi
            else:
                stock_holdings[i] = stock_holdings[i-1]
                option_U_holdings[i] = option_U_holdings[i-1]

            # Portfolio value: stock + hedging option + cash
            # Note: we are short V, so V does not appear here — it's the liability we're tracking against
            portfolio_value[i] = (stock_holdings[i] * S[i]
                                + option_U_holdings[i] * U_i
                                + cash[i])

        return t_grid, S, v, deltas, phis, portfolio_value
    
    def CV_delta_vega_hedging(self, simulator, N_paths, seed, rehedge_steps, b=1.0, fd_bump=1e-4):
        """
        Simulates a delta-vega-hedged path for a Short Call position.
        Implements the full Heston hedging portfolio: Pi = V + Delta*S + phi*U

        Uses a control variate correction on the Heston delta:
            Delta_CV = Delta_Heston_FD - b * (Delta_BS_FD - Delta_BS_analytical)
        where FD denotes a finite-difference (MC-noisy) estimate and analytical
        is the closed-form BS delta. This reduces variance in the delta estimate.

        Args:
            rehedge_steps (int): Number of time steps between rebalancing.
            b (float): Control variate coefficient. b=1 is the standard choice.
            fd_bump (float): Relative bump size h = fd_bump * S for finite differences.
        """
        # 1. Simulate the Real World Path (Heston dynamics)
        t_grid, S, v = simulator.simulate_path(seed=seed)

        # 2. Initialize arrays
        portfolio_value   = np.zeros(len(t_grid))
        deltas            = np.zeros(len(t_grid))
        vegas_V           = np.zeros(len(t_grid))
        vegas_U           = np.zeros(len(t_grid))
        phis              = np.zeros(len(t_grid))
        stock_holdings    = np.zeros(len(t_grid))
        option_U_holdings = np.zeros(len(t_grid))
        cash              = np.zeros(len(t_grid))

        # Helper: compute control-variate corrected Heston delta at a given node
        def compute_cv_delta(S_i, tau_i, v_i):
            h = fd_bump * S_i  # absolute bump size scales with S

            # --- Finite-difference Heston delta ---
            # Fixed this b/c originally just called simulator.estimate_delta_fd 
            # but there was issues with internal params being dynamically modified and not resetted after each call
            # 1. SAVE the original simulator state
            orig_tau = simulator.params.tau
            orig_S0 = simulator.params.S0
            orig_v0 = simulator.params.v0

            # --- Finite-difference Heston delta ---
            delta_heston_finite_diff, error = simulator.estimate_delta_finite_diff(
                N_paths=N_paths, tau_i=tau_i, v_i=v_i, option_type="call", dS=h, seed=seed
            )

            # 2. RESTORE the simulator state 
            simulator.params.tau = orig_tau
            if orig_S0 is not None: simulator.params.S0 = orig_S0
            if orig_v0 is not None: simulator.params.v0 = orig_v0

            # --- Finite-difference BS delta (same bump, same spot) ---
            V_bs_up   = simulator.get_bs_price(S_i + h, tau_i, v_i)
            V_bs_down = simulator.get_bs_price(S_i - h, tau_i, v_i)
            delta_bs_finite_diff = (V_bs_up - V_bs_down) / (2 * h)

            # --- Analytical BS delta ---
            delta_bs_analytical = simulator.get_bs_delta(S_i, tau_i)

            # --- Control variate correction ---
            return delta_heston_finite_diff - b * (delta_bs_finite_diff - delta_bs_analytical)

        # 3. Initial Setup (t=0)
        tau = simulator.params.tau

        V0 = simulator.get_bs_price(S[0], tau, v[0], strike = None)
        vegas_V[0] = simulator.get_bs_vega(S[0], tau)

        U0 = simulator.get_bs_price_U(S[0], tau, v[0])
        vegas_U[0] = simulator.get_bs_vega_U(S[0], tau)

        if abs(vegas_V[0]) == 0 or abs(-vegas_V[0] / vegas_U[0]) > 100:
            phis[0] = 0
        else:
            phis[0] = -vegas_V[0] / vegas_U[0]
        

        # Use CV-corrected delta from the start
        deltas[0] = compute_cv_delta(S[0], tau, v[0])
        stock_holdings[0] = deltas[0]
        option_U_holdings[0] = phis[0]

        cash[0] = V0 - (stock_holdings[0] * S[0]) - (option_U_holdings[0] * U0)

        # 4. Step through time
        for i in range(1, len(t_grid)):
            dt  = t_grid[i] - t_grid[i-1]
            #tau = simulator.params.tau - t_grid[i]
            tau = max(simulator.params.tau - t_grid[i], 1e-6)

            # Accrue interest on cash
            cash[i] = cash[i-1] * np.exp(simulator.params.r * dt)

            # Current prices and Greeks
            V_i = simulator.get_bs_price(S[i], tau, v[i], strike = None)
            U_i = simulator.get_bs_price_U(S[i], tau, v[i])

            current_vega_V = simulator.get_bs_vega(S[i], tau, vol_proxy=None, strike = None)
            current_vega_U = simulator.get_bs_vega_U(S[i], tau,vol_proxy=None)

            if abs(current_vega_U) == 0 or abs(-current_vega_V / current_vega_U) > 100:
                current_phi = phis[i-1]
            else:
                current_phi = -current_vega_V / current_vega_U

            # CV-corrected Heston delta
            current_delta = compute_cv_delta(S[i], tau, v[i])
            current_delta = np.clip(current_delta, -1.0, 1.0)

            deltas[i]  = current_delta
            vegas_V[i] = current_vega_V
            vegas_U[i] = current_vega_U
            phis[i]    = current_phi

            # Rehedge logic
            if i % rehedge_steps == 0 and i < len(t_grid) - 1:
                shares_to_trade = current_delta - stock_holdings[i-1]
                cash[i] -= shares_to_trade * S[i]
                stock_holdings[i] = current_delta

                options_to_trade = current_phi - option_U_holdings[i-1]
                cash[i] -= options_to_trade * U_i
                option_U_holdings[i] = current_phi
            else:
                stock_holdings[i]     = stock_holdings[i-1]
                option_U_holdings[i]  = option_U_holdings[i-1]

            portfolio_value[i] = (stock_holdings[i] * S[i]
                                + option_U_holdings[i] * U_i
                                + cash[i])

        return t_grid, S, v, deltas, phis, portfolio_value
    
    def apply_pls_filter(self, deltas: np.ndarray, lam: float) -> np.ndarray:
        """
        Applies a Penalized Least Squares filter to a 1D array.
        """
        N = len(deltas)
        if N <= 2:
            return deltas

        diags = np.array([1, -2, 1])
        D = sparse.diags(diags, [0, 1, 2], shape=(N-2, N), dtype=None)
        I = sparse.eye(N)
        A = I + lam * D.T.dot(D)
        A_csc = A.tocsc() # formatting for spsolve
        
        return spsolve(A_csc, deltas)
    
    def CV_delta_vega_hedging_pls(self, simulator, N_paths, seed, rehedge_steps, b=1.0, fd_bump=1e-4, pls_lambda = 100.0):
        # 1. Simulate the Real World Path
        t_grid, S, v = simulator.simulate_path(seed=seed)

        # 2. Initialize arrays
        portfolio_value   = np.zeros(len(t_grid))
        deltas            = np.zeros(len(t_grid))       # Will store the SMOOTHED deltas
        deltas_noisy      = np.zeros(len(t_grid))       # Will store the RAW CV deltas
        vegas_V           = np.zeros(len(t_grid))
        vegas_U           = np.zeros(len(t_grid))
        phis              = np.zeros(len(t_grid))
        stock_holdings    = np.zeros(len(t_grid))
        option_U_holdings = np.zeros(len(t_grid))
        cash              = np.zeros(len(t_grid))

        def compute_cv_delta(S_i, tau_i, v_i):
            h = fd_bump * S_i  

            # Save state
            orig_tau = simulator.params.tau
            orig_S0 = simulator.params.S0
            orig_v0 = simulator.params.v0

            delta_heston_finite_diff, _ = simulator.estimate_delta_finite_diff(
                N_paths=N_paths, tau_i=tau_i, v_i=v_i, option_type="call", dS=h, seed=seed
            )

            # Restore state
            simulator.params.tau = orig_tau
            if orig_S0 is not None: simulator.params.S0 = orig_S0
            if orig_v0 is not None: simulator.params.v0 = orig_v0

            V_bs_up   = simulator.get_bs_price(S_i + h, tau_i, v_i)
            V_bs_down = simulator.get_bs_price(S_i - h, tau_i, v_i)
            delta_bs_finite_diff = (V_bs_up - V_bs_down) / (2 * h)
            delta_bs_analytical = simulator.get_bs_delta(S_i, tau_i)

            return delta_heston_finite_diff - b * (delta_bs_finite_diff - delta_bs_analytical)

        # 3. Initial Setup (t=0)
        tau = simulator.params.tau

        V0 = simulator.get_bs_price(S[0], tau, v[0], strike=None)
        vegas_V[0] = simulator.get_bs_vega(S[0], tau)
        U0 = simulator.get_bs_price_U(S[0], tau, v[0])
        vegas_U[0] = simulator.get_bs_vega_U(S[0], tau)

        if abs(vegas_V[0]) == 0 or abs(-vegas_V[0] / vegas_U[0]) > 100:
            phis[0] = 0
        else:
            phis[0] = -vegas_V[0] / vegas_U[0]
        
        # At t=0, noisy and smoothed delta are the same
        initial_delta = compute_cv_delta(S[0], tau, v[0])
        deltas_noisy[0] = initial_delta
        deltas[0] = initial_delta
        
        stock_holdings[0] = deltas[0]
        option_U_holdings[0] = phis[0]
        cash[0] = V0 - (stock_holdings[0] * S[0]) - (option_U_holdings[0] * U0)

        # 4. Step through time
        for i in range(1, len(t_grid)):
            dt  = t_grid[i] - t_grid[i-1]
            tau = max(simulator.params.tau - t_grid[i], 1e-6)

            cash[i] = cash[i-1] * np.exp(simulator.params.r * dt)

            V_i = simulator.get_bs_price(S[i], tau, v[i], strike=None)
            U_i = simulator.get_bs_price_U(S[i], tau, v[i])

            current_vega_V = simulator.get_bs_vega(S[i], tau, vol_proxy=None, strike=None)
            current_vega_U = simulator.get_bs_vega_U(S[i], tau, vol_proxy=None)

            if abs(current_vega_U) == 0 or abs(-current_vega_V / current_vega_U) > 100:
                current_phi = phis[i-1]
            else:
                current_phi = -current_vega_V / current_vega_U

            # -------------------------------------------------------------
            # PLS Causal Filtering Logic
            # -------------------------------------------------------------
            
            # 1. Compute and store the raw, noisy CV delta
            raw_cv_delta = compute_cv_delta(S[i], tau, v[i])
            raw_cv_delta = np.clip(raw_cv_delta, -1.0, 1.0)
            deltas_noisy[i] = raw_cv_delta
            
            # 2. Rehedge logic with PLS smoothing
            if i % rehedge_steps == 0 and i < len(t_grid) - 1:
                
                # Apply PLS filter causally: only use data up to current step i
                if i >= 2:
                    smoothed_history = self.apply_pls_filter(deltas_noisy[:i+1], lam=pls_lambda)
                    current_delta = smoothed_history[-1] # Extract the causal estimate for time i
                else:
                    current_delta = deltas_noisy[i] # Not enough data to smooth yet
                
                # Store the smoothed delta
                deltas[i] = current_delta
                
                # Trade based on the SMOOTHED delta
                shares_to_trade = current_delta - stock_holdings[i-1]
                cash[i] -= shares_to_trade * S[i]
                stock_holdings[i] = current_delta

                options_to_trade = current_phi - option_U_holdings[i-1]
                cash[i] -= options_to_trade * U_i
                option_U_holdings[i] = current_phi
            else:
                # If not rehedging, just carry over previous positions
                deltas[i] = deltas[i-1]
                stock_holdings[i] = stock_holdings[i-1]
                option_U_holdings[i] = option_U_holdings[i-1]

            portfolio_value[i] = (stock_holdings[i] * S[i]
                                + option_U_holdings[i] * U_i
                                + cash[i])
            vegas_V[i] = current_vega_V
            vegas_U[i] = current_vega_U
            phis[i]    = current_phi

        return t_grid, S, v, deltas, phis, portfolio_value