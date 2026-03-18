import numpy as np
import time
from datetime import timedelta
from dataclasses import dataclass


from Heston_MC_class import HestonMonteCarlo
from Heston_params_class import HestonParams
from Heston_plots_class import HestonPlots
from Heston_Hedging_class import HestonHedging
from analytic_soln import analytic_heston_delta

def get_params_user_input():
    fields = {
        "S0": "Initial stock price",
        "K": "Strike price",
        "K_U": "Strike price for hedging option",
        "r": "Risk-free rate",
        "q": "Dividend yield",
        "v0": "Initial variance",
        "kappa": "Mean reversion speed",
        "theta": "Long-term variance",
        "sigma": "Volatility of variance",
        "rho": "Correlation",
        "tau": "Time to maturity"
    }
    
    data = {}
    for key, label in fields.items():
        while True:
            try:
                user_input = input(f"{label} ({key}): ")
                data[key] = float(user_input)
                break
            except ValueError:
                print(f'Error: {user_input} is not a valid number. Please try again: ')

    # Unpack the dictionary into your class
    return HestonParams(**data, analytical_delta=0.0)

def order_stat_CI(deltas, confidence_level):
    """
    Dynamically calculates the Empirical and Median confidence intervals 
    for an array of simulated Deltas using order statistics.
    """
    n = len(deltas)
    deltas_sorted = np.sort(deltas)
    
    # Calculate alpha for the tails
    alpha = 1.0 - confidence_level
    
    # Calculate the exact indices for the bottom and top percentiles
    lower_emp_idx = int(np.floor(n * (alpha / 2)))
    upper_emp_idx = int(np.ceil(n * (1 - alpha / 2))) - 1
    
    # Use np.clip to prevent out-of-bounds errors if N is extremely small
    lower_emp_idx = np.clip(lower_emp_idx, 0, n - 1)
    upper_emp_idx = np.clip(upper_emp_idx, 0, n - 1)
    
    emp_ci = (round(deltas_sorted[lower_emp_idx],4), round(deltas_sorted[upper_emp_idx],4))
    return emp_ci

if __name__ == "__main__":      
    # Set up parameters
    params = HestonParams(
        S0=100.0, K=100.0, K_U = 110.0, r=0.03, q=0.0, v0=0.05, lambd = 0, kappa=5.0, 
        theta=0.05, sigma=0.5, rho=-0.8, tau=0.5, analytical_delta = 0
    )
    #params = get_params_user_input()
    true_delta = analytic_heston_delta(params.S0,params.K,params.r,params.v0,params.lambd,params.kappa,
                                       params.theta,params.sigma,params.rho,params.tau,params.q)
    params.analytical_delta = true_delta

    print("\nModel Parameters:")
    for field in params.__dataclass_fields__:
        value = getattr(params, field)
        print(f"  {field:10s} = {value}")
    
    # instance of simulator
    simulator = HestonMonteCarlo(params, N_steps=100)
    # instance of technique 
    model = HestonHedging()
    # Simulate and plot a few paths
    print("\n" + "-"*80)
    print("Simulating sample paths...")
    print("-"*80)
    
    plotter = HestonPlots()
    plotter.plot_paths(simulator,N_paths=20, seed=42)

    # Price options
    print("\n" + "-"*80)
    print("Monte Carlo Pricing")
    print("-"*80)
    
    n_paths_per_sim = 100
    n_sims = 10
    paths_to_print = np.logspace(np.log10(10), np.log10(1000), num=6, dtype=int)
    #print(f"\nUsing {N_paths:,} paths...")
    
    # _,call_price, call_std = simulator.price_option(N_paths, 'call', seed=42)
    # print(f"\nCall Price: ${call_price:.4f} ± ${1.96*call_std:.4f} (95% CI)")
    
    # _, put_price, put_std = simulator.price_option(N_paths, 'put', seed=42)
    # print(f"Put Price:  ${put_price:.4f} ± ${1.96*put_std:.4f} (95% CI)")
    
    # # Check put-call parity
    # pcp_lhs = call_price - put_price
    # pcp_rhs = params.S0 * np.exp(-params.q * params.tau) - params.K * np.exp(-params.r * params.tau)
    # print(f"\nPut-Call Parity Check:")
    # print(f"  C - P = ${pcp_lhs:.4f}")
    # print(f"  S·exp(-q·T) - K·exp(-r·T) = ${pcp_rhs:.4f}")
    # print(f"  Difference: ${abs(pcp_lhs - pcp_rhs):.4f}")
    
    print(f"\nSummary statistics of generated Heston MC deltas with {n_sims:,} simulations, each with {n_paths_per_sim} paths, with 100 time steps per path...\n")
   
    ave_deltas_across_sims = []
    start = time.time()

    for seed in range(n_sims):
        delta, delta_stde = simulator.estimate_delta_finite_diff(
                                N_paths_per_sim= n_paths_per_sim, 
                                tau_i=simulator.params.tau, 
                                v_i=simulator.params.v0, 
                                option_type='call', 
                                dS=0.01, 
                                seed = seed
        )
        ave_deltas_across_sims.append(delta)    

    elapsed = time.time() - start
    formatted_time = str(timedelta(seconds=int(elapsed)))
    
    # Calculate CI with order statistics to avoid assuming normality of distribution across simulations
    conf_level = 0.95
    conf_interval = order_stat_CI(ave_deltas_across_sims,conf_level)
    mean_delta = np.mean(ave_deltas_across_sims)

    print(f"  Analytical Delta:     {simulator.params.analytical_delta:.6f}")
    print(f"  Monte Carlo Delta:    {mean_delta:.6f}")
    print(f"  {int(conf_level*100)}% Conf Interval:    {conf_interval}")
    print(f"  Absolute Error:       {abs(mean_delta - simulator.params.analytical_delta)*100:.2f}%")
    print(f"  Relative Error:       {abs(mean_delta - simulator.params.analytical_delta)/simulator.params.analytical_delta*100:.2f}%")
    print(f"  Computation Time:     {formatted_time}")
    
    # Convergence study
    print("\n" + "-"*80)
    num_sims = 1000
    print(f"One simulation of Monte Carlo Estimation of delta, each with {n_paths_per_sim} paths...")
    print("-"*80)

    #regular_mc = 1
    #control_variate = 2
    #penalized_least_squares = 3
    seeds_visual = [x for x in range(1, 21)]

    # confirming convergence of Delta for one simulation
    #plotter.convergence_study(simulator, n_paths_per_sim, paths_to_print, simulator.params.analytical_delta)

    print("\n" + "-"*80)
    print(f"Summary statistics on {n_sims} number of simulations of {n_paths_per_sim} paths each with techniques:")
    print("-"*80)
    
    for i in range(3):
        plotter.plot_mc_delta_estimates(simulator, model, n_sims, n_paths_per_sim, seed=42, tech=i+1)

    print("\n" + "-"*80)
    print(f"Plotting trajectory of portfolio with {len(seeds_visual)} number of simulations of {n_paths_per_sim} paths each with techniques:")
    print("-"*80)    
    # for i in range(3):
    #     plotter.plot_hedging_trajectory(simulator,model,n_paths_per_sim,seeds = seeds,tech=i+1)
    plotter.plot_hedging_trajectory(simulator,model,n_paths_per_sim,seeds = seeds_visual,tech=3)



