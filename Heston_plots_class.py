import numpy as np
import matplotlib.pyplot as plt
from typing import List
import time
from dataclasses import dataclass

from Heston_MC_class import HestonMonteCarlo
from Heston_Hedging_class import HestonHedging

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
        plt.savefig('./plots_CV/Stock_Var_Plot.png', dpi=150)
        print("Saved trajectory plots of Stock price and volatility as Stock_Var_Plot.png")


    def convergence_study(self,simulator, path_counts, analytical_delta, seed, control_var):
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
            delta, _ = simulator.estimate_delta_finite_diff(N, tau_i = simulator.params.tau, v_i = simulator.params.v0, option_type = 'call', dS=0.01, seed=seed)
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
        if(control_var):
            plt.savefig('./plots_CV/Delta_Convergence_CV.png')
            print("Saved convergence of Delta estimates with control variates as Delta_Convergence_CV.png")
        else:
            plt.savefig('./plots/Delta_Convergence.png')
            print("Saved convergence of Delta estimates as Delta_Convergence.png")


    def plot_hedging_trajectory(self, simulator, technique: HestonHedging, N_paths, seed, control_var):
        seeds = [x for x in range(1, 20)]
        
        fig, axes = plt.subplots(5, 1, figsize=(12, 18))
        ax2, ax3, ax4, ax5, ax6 = axes
        for seed in seeds:
            if(control_var):
                t, S, v, deltas, phis, port_val = technique.CV_delta_vega_hedging(
                simulator, N_paths, seed=seed, rehedge_steps=1
                )
            else:
                t, S, v, deltas, phis, port_val = technique.delta_vega_hedging(
                    simulator, seed=seed, rehedge_steps=1
                )
            # Compute portfolio components at each timestep
            V = np.array([
                simulator.get_bs_price(S[i], simulator.params.tau - t[i], vol_proxy=np.sqrt(v[i]), strike = None)
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
            #ax1.plot(t, S, alpha=alpha, linewidth=lw)
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
        # ax1.axhline(simulator.params.K, color='r', linestyle='--', label='Strike K')
        # ax1.set_ylabel('Stock Price $S$')
        # ax1.set_title('Underlying Asset Path')
        # ax1.legend(fontsize=8)
        # ax1.grid(True, alpha=0.3)

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
        if(control_var): 
            plt.savefig('./plots_CV/Portfolio_component_CV.png', dpi=150)
            print("Saved trajectory plots with control variates as Portfolio_component_CV.png")
        else: 
            plt.savefig('./plots/Portfolio_component.png', dpi=150)
            print("Saved trajectory plots as Portfolio_component.png")