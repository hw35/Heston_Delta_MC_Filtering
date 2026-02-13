import numpy as np
import time
from dataclasses import dataclass


from Heston_MC_class import HestonMonteCarlo
from Heston_params_class import HestonParams
from Heston_plots_class import HestonPlots
from analytic_soln import analytic_heston_delta

def get_params_user_input():
    fields = {
        "S0": "Initial stock price",
        "K": "Strike price",
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

if __name__ == "__main__":      
    # Set up parameters
    params = HestonParams(
        S0=100.0, K=100.0, r=0.03, q=0.0, v0=0.05, lambd = 0, kappa=5.0, 
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
    
    # Create simulator
    simulator = HestonMonteCarlo(params, N_steps=100)
    
    # Simulate and plot a few paths
    print("\n" + "-"*80)
    print("Simulating sample paths...")
    print("-"*80)
    
    plotter = HestonPlots()
    plotter.plot_paths(simulator,N_paths=20, seed=7)

    # Price options
    print("\n" + "-"*80)
    print("Monte Carlo Pricing")
    print("-"*80)
    
    N_paths = 10000
    print(f"\nUsing {N_paths:,} paths...")
    
    _,call_price, call_std = simulator.price_option(N_paths, 'call', seed=7)
    print(f"\nCall Price: ${call_price:.4f} ± ${1.96*call_std:.4f} (95% CI)")
    
    _, put_price, put_std = simulator.price_option(N_paths, 'put', seed=7)
    print(f"Put Price:  ${put_price:.4f} ± ${1.96*put_std:.4f} (95% CI)")
    
    # Check put-call parity
    pcp_lhs = call_price - put_price
    pcp_rhs = params.S0 * np.exp(-params.q * params.tau) - params.K * np.exp(-params.r * params.tau)
    print(f"\nPut-Call Parity Check:")
    print(f"  C - P = ${pcp_lhs:.4f}")
    print(f"  S·exp(-q·T) - K·exp(-r·T) = ${pcp_rhs:.4f}")
    print(f"  Difference: ${abs(pcp_lhs - pcp_rhs):.4f}")
    
    # Estimate delta
    print("\n" + "-"*80)
    print("Delta Estimation (Finite Differences)")
    print("-"*80)
    
    print(f"\nEstimating delta with {N_paths:,} paths...")
    start = time.time()
    delta, delta_std = simulator.estimate_delta_finite_diff(N_paths, 'call', dS=0.01, seed=7)
    elapsed = time.time() - start
    
    print(f"\nResults:")
    print(f"  Analytical Delta:    {simulator.params.analytical_delta:.6f}")
    print(f"  Monte Carlo Delta:   {delta:.6f} ± {1.96*delta_std:.6f}")
    print(f"  Absolute Error:      {abs(delta - simulator.params.analytical_delta):.6f}")
    print(f"  Relative Error:      {abs(delta - simulator.params.analytical_delta)/simulator.params.analytical_delta*100:.2f}%")
    print(f"  Computation Time:    {elapsed:.2f} seconds")
    
    # Convergence study
    print("\n" + "-"*80)
    print("Convergence Study")
    print("-"*80)
    
    path_counts = [100, 500, 1000, 2500, 5000, 10000]
    plotter.convergence_study(simulator,path_counts, simulator.params.analytical_delta, seed=7)
    
    plotter.plot_hedging_trajectory(simulator,seed=123)

