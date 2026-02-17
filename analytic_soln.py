import numpy as np
from scipy.integrate import quad

def analytic_heston_delta(S,K,r,v0,lambd,kappa,theta,sigma,rho,tau,q):
    P1 = analytic_heston_prob(S,K,r,v0,lambd,kappa,theta,sigma,rho,tau,True,q)
    delta = np.exp(-q*tau)*P1
    print("\n" + "-"*80)
    print("Analytic Delta is ", delta)
    print("-"*80)
    return delta

def analytic_heston_prob(S,K,r,v0,lambd,kappa,theta,sigma,rho,tau,isP1,q):
    def integrand(phi):
        return analytic_heston_chf(S,K,r,v0,lambd,kappa,theta,sigma,rho,tau,isP1,phi,q)
    value = quad(integrand,0,1000)[0]
    prob = 0.5 + (1.0/np.pi) * value
    return prob

def analytic_heston_chf(S,K,r,v0,lambd,kappa,theta,sigma,rho,tau,isP1,phi,q):
    x = np.log(S)
    a = kappa * theta
    if(isP1):
        b = kappa + lambd - rho*sigma
        u = 0.5
    else:
        b = kappa + lambd
        u = -0.5
    Q_val = b - rho*sigma*1j*phi
    d = np.sqrt((-Q_val)**2 - sigma**2*(2*u*1j*phi-phi**2))    
    g = (Q_val+d) / (Q_val-d)
    c = 1.0/g # For Little Trap
    G_val = (1 - c*np.exp(-d*tau))/(1-c)
    D = (Q_val-d) / (sigma**2) * (1-np.exp(-d*tau))/(1-c*np.exp(-d*tau))
    C = (r-q)*1j*phi*tau + a/(sigma**2) * ((Q_val-d)*tau-2*np.log(G_val))

    f = np.exp(C+D*v0+1j*phi*x)
    integ = ((np.exp(-1j*phi*np.log(K))*f)/(1j*phi)).real
    return integ