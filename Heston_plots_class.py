import numpy as np
import matplotlib.pyplot as plt
from typing import List
import time
from datetime import timedelta
from dataclasses import dataclass

from Heston_MC_class import HestonMonteCarlo
from Heston_Hedging_class import HestonHedging
from analytic_soln import analytic_stock_delta

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


    def convergence_study(self,simulator, n_paths_per_sim, paths_to_print, analytical_delta):
        """
        Study convergence of Monte Carlo delta estimate.
        """
        deltas = []
        errors = []
        times = []
        
        print(f"\n{'N_paths':<12} {'Delta':<12} {'Error':<12} {'Time (s)':<12}")
        print("-" * 50)
        
        for N in paths_to_print:
            start = time.time()
            delta, _ = simulator.estimate_delta_finite_diff(N, tau_i = simulator.params.tau, v_i = simulator.params.v0, option_type = 'call', dS=0.01, seed = 42)
            elapsed = time.time() - start
            formatted_time = str(timedelta(seconds=int(elapsed)))

            deltas.append(delta)
            error = abs(delta - analytical_delta)
            errors.append(error)
            times.append(elapsed)
            print(f"{N:<12,} {delta:<12.6f} {error:<12.6f} {formatted_time}")
        
        # Plot convergence
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Delta convergence
        ax1.semilogx(paths_to_print, deltas, 'bo-', label='MC Estimate', linewidth=2, markersize=8)
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
        print("Saved convergence of Delta estimates from a singular simulation as Delta_Convergence.png")

    def plot_mc_delta_estimates(self, simulator, model, n_sims, n_paths_per_sim, seed, tech, b=1.0, fd_bump=1e-4, pls_lambda=100.0):
        """
        Plots the initial Delta estimates across different seeds and calculates 
        estimator variance and relative error.
        """
        ave_deltas_across_sims = []
        deltas_across_sims = [[]]
        start = time.time()

        #true_stock_deltas_over_time = model.analytical_delta_vega_hedging(simulator,seed)
        #ave_true_delta = np.mean(true_stock_deltas_over_time)

        true_stock_delta = analytic_stock_delta()

        # Execute the correct technique based on the 'tech' parameter
        for seed in range(n_sims):
            deltas = []
            beta_CV = 0
            if tech == 1:
                _, _, _, deltas, _, _, beta_CV = model.delta_vega_hedging(
                    simulator, n_paths_per_sim, seed, rehedge_steps=1, fd_bump=fd_bump
                )
            elif tech == 2:
                _, _, _, deltas, _, _ = model.CV_delta_vega_hedging(
                    simulator, n_paths_per_sim, seed, beta_CV, rehedge_steps=1, b=b, fd_bump=fd_bump
                )
            else:
                _, _, _, deltas, _, _ = model.CV_delta_vega_hedging_pls(
                    simulator, n_paths_per_sim, seed, rehedge_steps=1, b=b, fd_bump=fd_bump, pls_lambda=pls_lambda
                )
            ave_delta = np.mean(deltas)
            ave_deltas_across_sims.append(ave_delta)
            deltas_across_sims.append(deltas)

        elapsed = time.time() - start
        formatted_time = str(timedelta(seconds=int(elapsed)))

        
        # Convert to numpy array for analytics
        #print(deltas)
        ave_deltas_across_sims = np.array(ave_deltas_across_sims)
        #true_stock_deltas_over_time = np.array(true_stock_deltas_over_time)
        #true_delta = simulator.params.analytical_delta

        # --- 1. Plotting ---
        fig, ax = plt.subplots(figsize=(10, 6))
        #print(len(deltas))
        ax.plot(ave_deltas_across_sims, 'bo-', label=f'Average Delta from each Simulation with {n_paths_per_sim} paths', alpha=0.7)
        ax.axhline(y=true_stock_delta, color='r', linestyle='--', linewidth=2, label=f'Analytical = {true_stock_delta:.6f}')
        
        ax.set_xlabel('Simulation number')
        ax.set_ylabel('Delta Estimate $\Delta$')
        technique_name = {1: "Regular MC", 2: "Control Variate", 3: "CV + PLS"}[tech]
        ax.set_title(f'Monte Carlo Delta Estimates Across Simulations\n({technique_name})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        # --- 2. Analytics ---
        estimator_variance = np.var(ave_deltas_across_sims, ddof=1)*100
        relative_errors = np.abs((ave_deltas_across_sims - true_stock_delta) / true_stock_delta)
        
        mean_relative_error = np.mean(relative_errors) * 100 
        mean_delta = np.mean(ave_deltas_across_sims)

        average_bias = abs(np.mean(ave_deltas_across_sims) - true_stock_delta)
        relative_ave_bias = (average_bias / true_stock_delta) * 100

        tech_name_full = {1: "Regular MC estimate", 2: "Control Variate estimate", 3: "CV and PLS estimate"}[tech]

        print(f"\n{tech_name_full}")
        print(f"  Analytical Net Delta:      {true_stock_delta:.4f}")
        print(f"  Average Simulated Delta:   {mean_delta:.4f}")
        print(f"  Estimator Variance:        {estimator_variance:.4f}%") 
        print(f"  Average Relative Error:    {mean_relative_error:.4f}%")
        print(f"  Average Relative Bias:     {relative_ave_bias:.4f}%")
        print(f"  Computation Time:          {formatted_time}")

        if(tech == 1):
            plt.savefig('./plots/Delta_Estimates.png', dpi=150)
            print("Saved Delta Convergence plots for regular MC as Delta_Estimates.png")
        elif(tech == 2): 
            plt.savefig('./plots_CV/Delta_Estimates_CV.png', dpi=150)
            print("Saved Delta Convergence plots for Control Variate as Delta_Estimates_CV.png")
        else: 
            plt.savefig('./plots_PLS/Delta_Estimates_PLS.png', dpi=150)
            print("Saved Delta Convergence plots for PLS as Delta_Estimates_PLS.png")

    # def plot_mc_delta_estimates(self, simulator, model, n_sims, n_paths_per_sim, seeds, tech, b=1.0, fd_bump=1e-4, pls_lambda=100.0):
    #     """
    #     Plots the initial Delta estimates across different seeds and calculates 
    #     estimator variance and relative error.
    #     """
    #     deltas = []
    #     start = time.time()

    #     for seed in seeds:
    #         # Execute the correct technique based on the 'tech' parameter
    #         if tech == 1:
    #             _, _, _, d, _, _ = model.delta_vega_hedging(simulator, seed, rehedge_steps=1)
    #         elif tech == 2:
    #             _, _, _, d, _, _ = model.CV_delta_vega_hedging(simulator, n_sims, n_paths_per_sim, seed, rehedge_steps=1, b=b, fd_bump=fd_bump)
    #         else:
    #             _, _, _, d, _, _ = model.CV_delta_vega_hedging_pls(simulator, n_sims, n_paths_per_sim, seed, rehedge_steps=1, b=b, fd_bump=fd_bump, pls_lambda=pls_lambda)
            
    #         # Append the delta estimate at t=N_steps/2 for this specific seed
    #         ####
    #         deltas.append(d[-1])

    #     elapsed = time.time() - start
    #     formatted_time = str(timedelta(seconds=int(elapsed)))

        
    #     # Convert to numpy array for analytics
    #     deltas = np.array(deltas)
    #     last_delta = deltas[-1]
    #     true_delta = simulator.params.analytical_delta

    #     # --- 1. Plotting ---
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #     ax.plot(seeds, deltas, 'bo-', label='Simulated Delta at $t=0$', alpha=0.7)
    #     ax.axhline(y=true_delta, color='r', linestyle='--', linewidth=2, label=f'Analytical = {true_delta:.6f}')
        
    #     ax.set_xlabel('Seed')
    #     ax.set_ylabel('Delta Estimate $\Delta$')
    #     technique_name = {1: "Regular MC", 2: "Control Variate", 3: "CV + PLS"}[tech]
    #     ax.set_title(f'Monte Carlo Delta Estimates Across Independent Seeds\n({technique_name})')
    #     ax.grid(True, alpha=0.3)
    #     ax.legend()
    #     plt.tight_layout()

    #     # --- 2. Analytics ---
    #     estimator_variance = np.var(deltas, ddof=1)*100
    #     relative_errors = np.abs((deltas - true_delta) / true_delta)
        
    #     mean_relative_error = np.mean(relative_errors) * 100 
    #     mean_delta = np.mean(deltas)

    #     average_bias = abs(np.mean(deltas) - true_delta)
    #     relative_ave_bias = (average_bias / true_delta) * 100

    #     tech_name_full = {1: "Regular MC estimate", 2: "Control Variate estimate", 3: "CV and PLS estimate"}[tech]

    #     print(f"\n{tech_name_full}")
    #     print(f"  Analytical Delta:          {true_delta:.6f}")
    #     print(f"  Average Simulated Delta:   {mean_delta:.6f}")
    #     print(f"  Estimator Variance:        {estimator_variance:.6f}%") 
    #     print(f"  Average Relative Error:    {mean_relative_error:.6f}%")
    #     print(f"  Average Relative Bias:     {relative_ave_bias:.6f}%")
    #     print(f"  Computation Time:          {formatted_time}")

    #     if(tech == 1):
    #         plt.savefig('./plots/Delta_Estimates.png', dpi=150)
    #         print("Saved Delta Convergence plots for regular MC as Delta_Estimates.png")
    #     elif(tech == 2): 
    #         plt.savefig('./plots_CV/Delta_Estimates_CV.png', dpi=150)
    #         print("Saved Delta Convergence plots for Control Variate as Delta_Estimates_CV.png")
    #     else: 
    #         plt.savefig('./plots_PLS/Delta_Estimates_PLS.png', dpi=150)
    #         print("Saved Delta Convergence plots for PLS as Delta_Estimates_PLS.png")

    def plot_hedging_trajectory(self, simulator, model: HestonHedging, num_paths_per_sim, seeds, tech):
        
        fig, axes = plt.subplots(5, 1, figsize=(12, 18))
        ax2, ax3, ax4, ax5, ax6 = axes
        for seed in seeds:
            if(tech == 1):
                t, S, v, deltas, phis, port_val = model.delta_vega_hedging(
                    simulator, num_paths_per_sim, seed=seed, rehedge_steps=1
                )
            elif(tech==2):
                t, S, v, deltas, phis, port_val = model.CV_delta_vega_hedging(
                    simulator, num_paths_per_sim, seed=seed, rehedge_steps=1
                )
            else:
                t, S, v, deltas, phis, port_val = model.CV_delta_vega_hedging_pls(
                    simulator, num_paths_per_sim, seed=seed, rehedge_steps=1
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
        if(tech == 1):
            plt.savefig('./plots/Portfolio_component.png', dpi=150)
            print("Saved trajectory plots as Portfolio_component.png")
        elif(tech == 2): 
            plt.savefig('./plots_CV/Portfolio_component_CV.png', dpi=150)
            print("Saved trajectory plots with control variates as Portfolio_component_CV.png")
        else: 
            plt.savefig('./plots_PLS/Portfolio_component_PLS.png', dpi=150)
            print("Saved trajectory plots with PLS smoother as Portfolio_component_PLS.png")