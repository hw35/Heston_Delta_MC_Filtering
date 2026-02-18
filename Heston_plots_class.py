import numpy as np
import matplotlib.pyplot as plt
from typing import List
import time
from dataclasses import dataclass

from Heston_MC_class import HestonMonteCarlo

@dataclass
class HestonPlots:

    def plot_paths(self,simulator: HestonMonteCarlo, N_paths, seed: int = 42):
        """
        Plot sample paths of stock price and variance.
        """
        np.random.seed(seed)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
        for i in range(N_paths):
            t_grid, S_path, v_path = simulator.simulate_path()
            ax1.plot(t_grid, S_path, alpha=0.6, linewidth=0.8)
            ax2.plot(t_grid, v_path, alpha=0.6, linewidth=0.8)
        
        ax1.axhline(y=simulator.params.K, color='r', linestyle='--', label='Strike', linewidth=2)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Stock Price')
        ax1.set_title(f'Simulated Stock Price Paths (N={N_paths})')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.axhline(y=simulator.params.theta, color='r', linestyle='--', 
                    label=f'Long-term variance θ={simulator.params.theta}', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Variance')
        ax2.set_title(f'Simulated Variance Paths (N={N_paths})')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plt.tight_layout()
        plt.savefig('./plots/Stock_Var_Plot.png', dpi=150)
        print("Saved trajectory plots of Stock price and volatility as Stock_Var_Plot.png")


    def convergence_study(self,simulator, path_counts, analytical_delta, seed):
        """
        Study convergence of Monte Carlo delta estimate.
        """
        deltas = []
        errors = []
        times = []
        
        print(f"\n{'N_paths':<12} {'Delta':<12} {'Error':<12} {'Time (s)':<12}")
        print("-" * 50)
        
        for N in path_counts:
            start = time.time()
            delta, _ = simulator.estimate_delta_finite_diff(N, 'call', dS=0.01, seed=seed)
            elapsed = time.time() - start
            
            deltas.append(delta)
            error = abs(delta - analytical_delta)
            errors.append(error)
            times.append(elapsed)
            
            print(f"{N:<12,} {delta:<12.6f} {error:<12.6f} {elapsed:<12.2f}")
        
        # Plot convergence
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Delta convergence
        ax1.semilogx(path_counts, deltas, 'bo-', label='MC Estimate', linewidth=2, markersize=8)
        ax1.axhline(y=analytical_delta, color='r', linestyle='--', 
                    label=f'Analytical = {analytical_delta:.6f}', linewidth=2)
        ax1.set_xlabel('Number of Paths')
        ax1.set_ylabel('Delta')
        ax1.set_title('Delta Convergence')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Error vs time
        ax2.loglog(times, errors, 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('Computation Time (seconds)')
        ax2.set_ylabel('Absolute Error')
        ax2.set_title('Accuracy vs Computational Cost')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./plots/Delta_Convergence.png')
        print("Saved convergence of Delta estimates as Delta_Convergence.png")

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

    def simulate_delta_vega_hedging(self, simulator, seed, rehedge_steps):
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

    def plot_hedging_trajectory(self, simulator, seed):
        seeds = [x for x in range(1, 20)]
        
        fig, axes = plt.subplots(6, 1, figsize=(12, 22))
        ax1, ax2, ax3, ax4, ax5, ax6 = axes

        for seed in seeds:
            t, S, v, deltas, phis, port_val = self.simulate_delta_vega_hedging(
                simulator, seed=seed, rehedge_steps=1
            )
            # Compute portfolio components at each timestep
            V = np.array([
                simulator.get_bs_price(S[i], simulator.params.tau - t[i], vol_proxy=np.sqrt(v[i]))
                for i in range(len(t))
            ])
            delta_S = deltas * S  # Delta * Stock price

            U = np.array([
                simulator.get_bs_price_U(S[i], simulator.params.tau - t[i], vol_proxy=np.sqrt(v[i]))
                for i in range(len(t))
            ])
            phi_U = phis * U  # phi * hedging option price

            payoff = max(S[-1] - simulator.params.K, 0)
            final_error = port_val[-1] - payoff

            alpha = 0.5
            lw = 0.8

            # --- Plot each component ---
            ax1.plot(t, S, alpha=alpha, linewidth=lw)
            ax2.plot(t, V, alpha=alpha, linewidth=lw)
            ax3.plot(t, delta_S, alpha=alpha, linewidth=lw)
            ax4.plot(t, phi_U, alpha=alpha, linewidth=lw)
            ax5.plot(t, phis, alpha=alpha, linewidth=lw)
            ax6.plot(t, port_val, alpha=alpha, linewidth=lw)

            if seed in [1, 5, 10]:
                ax6.scatter(
                    [t[-1]], [payoff], color='r', zorder=5,
                    label=f'Payoff: {payoff:.2f} (seed {seed})'
                )

        # 1. Stock Price
        ax1.axhline(simulator.params.K, color='r', linestyle='--', label='Strike K')
        ax1.set_ylabel('Stock Price $S$')
        ax1.set_title('Underlying Asset Path')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. Option V (target option liability)
        ax2.set_ylabel('Value')
        ax2.set_title('Target Option Value $V(S, v, t)$')
        ax2.grid(True, alpha=0.3)

        # 3. Delta * S (stock hedge component)
        ax3.set_ylabel('Value')
        ax3.set_title('Stock Hedge Component $\\Delta \\cdot S$')
        ax3.grid(True, alpha=0.3)

        # 4. phi * U (volatility hedge component)
        ax4.set_ylabel('Value')
        ax4.set_title('Volatility Hedge Component $\\phi \\cdot U$')
        ax4.grid(True, alpha=0.3)

        # 5. phi (volatility hedge ratio)
        ax5.set_ylabel('$\\phi$')
        ax5.set_title('Volatility Hedge Ratio $\\phi = -\\mathcal{V}_V / \\mathcal{V}_U$')
        ax5.axhline(0, color='k', linestyle='--', linewidth=0.8)
        ax5.grid(True, alpha=0.3)

        # 6. Total portfolio value vs payoff
        ax6.set_ylabel('Value')
        ax6.set_xlabel('Time (Years)')
        ax6.set_title(f'Total Hedge Portfolio Value\nFinal Tracking Error (last seed): {final_error:.4f}')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('./plots/Portfolio_component.png', dpi=150)
        print("Saved trajectory plots as Portfolio_component.png")