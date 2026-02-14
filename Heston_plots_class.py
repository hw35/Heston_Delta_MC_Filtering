import numpy as np
import matplotlib.pyplot as plt
from typing import List
import time
from dataclasses import dataclass

from Heston_MC_class import HestonMonteCarlo

@dataclass
class HestonPlots:

    def plot_paths(self,simulator: HestonMonteCarlo, N_paths, seed: int = 7):
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
        plt.savefig('./plots/Stock_Var_Plot.png')
        print("Saved trajectory plots of Stock price and volatility as Stock_Var_Plot.png")


    def convergence_study(self,simulator: HestonMonteCarlo, path_counts: List[int], 
                        analytical_delta: float = 0.630402, seed: int = 7):
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

    def simulate_delta_hedging(self,simulator: HestonMonteCarlo, seed, rehedge_steps=1):
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

    def plot_hedging_trajectory(self,simulator: HestonMonteCarlo, seed=7):
    # Run simulation

        seeds = [x for x in range(1,20)]
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

        for seed in seeds:
            t, S, v, delta, port_val = self.simulate_delta_hedging(simulator,seed=seed, rehedge_steps=1)
        
            # Calculate Option Liability (Theoretical BS Price) for comparison
            liability = np.array([
                (simulator.get_bs_delta(s, simulator.params.tau - ti) * s) # Approximate
                for ti, s in zip(t, S) 
            ])
            # More accurate liability calculation requires pricing the option at every step
            # We will simply plot the Payoff at expiry to see the hedge error
            payoff = max(S[-1] - simulator.params.K, 0)
            final_error = port_val[-1] - payoff  
            ax1.plot(t,S, alpha = 0.6, linewidth = 0.8)
            ax2.plot(t, delta, alpha = 0.6, linewidth = 0.8)
            ax3.plot(t, port_val,alpha = 0.6, linewidth = 0.8)  
            if(seed == 1 or seed == 5 or seed == 10):
                ax3.scatter([t[-1]], [payoff], color='r', zorder=5, label=f'Option Liability (Payoff): {payoff:.2f}, seed: {seed}')  # FIXED

        # 1. Stock Price & Strike
        #ax1.plot(label='Heston Stock Path')
        ax1.axhline(simulator.params.K, color='r', linestyle='--', label='Strike')
        ax1.set_ylabel('Stock Price')
        ax1.set_title('Underlying Asset Path')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Delta Evolution
        #ax2.plot(t, delta, color='orange', label='Hedge Ratio (Delta)')
        ax2.axhline(simulator.params.analytical_delta, color='r', linestyle='--', label='Analytical Delta')
        ax2.set_ylabel('Delta')
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_title('Delta Hedging Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Hedge P&L (Cash + Stock)
        # At expiry, if we sell the stock and pay the option payoff, what is left?
        # Tracking (Stock Value + Cash Account)
        #ax3.plot(t, port_val, color='green', label='Hedge Portfolio Value')
        
        # Draw the option liability at the end 

        ax3.set_ylabel('Value')
        ax3.set_xlabel('Time (Years)')
        ax3.set_title(f'Hedge Portfolio vs Liability\nFinal Tracking Error: {final_error:.4f}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('./plots/Portfolio_component.png')
        print("Saved trajectory plots of portfolio component as Portfolio_component.png")