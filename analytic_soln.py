import numpy as np
from scipy.integrate import quad
from Heston_params_class import HestonParams

params = HestonParams(
    S0=100.0, K=100.0, K_U = 110.0, r=0.03, q=0.0, v0=0.05, lambd = 0, kappa=5.0, 
    theta=0.05, sigma=0.5, rho=-0.8, tau=0.5, analytical_delta = 0
)
def analytic_heston_delta(S,vi,tau,isP1,isU,q=params.q):
    if(isU): K_active = params.K_U
    else: K_active = params.K
    P1 = analytic_heston_prob(S=S,K=K_active,vi=vi,isP1=isP1,tau=tau,dPdv=False)
    delta = np.exp(-q*tau)*P1
    return delta

def analytic_vega(S,vi,tau,isU):
    K_active = 0
    if(isU): K_active = params.K_U
    else: K_active = params.K
    dP1dv = analytic_heston_prob(S,K_active,vi,tau,isP1=True,dPdv=True)
    dP2dv = analytic_heston_prob(S,K_active,vi,tau,isP1=False,dPdv=True)
    vega = S*dP1dv - K_active*np.exp(-params.q*tau)*dP2dv
    return vega

def analytic_varphi():
    vega_V = analytic_vega(params.S0,params.v0,params.tau,isU=False)
    vega_U = analytic_vega(params.S0,params.v0,params.tau,isU=True)
    return (-vega_V/vega_U)

def analytic_stock_delta():
    delta_V = analytic_heston_delta(S=params.S0,vi=params.v0,tau=params.tau,isU=False,isP1=True)
    delta_U = analytic_heston_delta(S=params.S0,vi=params.v0,tau=params.tau,isU=True,isP1=True)
    varphi = analytic_varphi()
    net_delta = delta_V - varphi * delta_U
    print(net_delta)
    return net_delta

def analytic_heston_prob(S,K,vi,tau,isP1,dPdv):
    def integrand(phi):
        return analytic_heston_chf(S=S,K=K,vi=vi,isP1=isP1,tau=tau,phi = phi,dPdv=dPdv)
    value = quad(integrand,0,1000)[0]
    prob = 0.5 + (1.0/np.pi) * value
    if(dPdv): prob -= 0.5
    return prob

def analytic_heston_chf(S,K,vi,tau,isP1,phi,dPdv,r=params.r,lambd=params.lambd,kappa = params.kappa,theta=params.theta,sigma=params.sigma,rho=params.rho,q=params.q):
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

    f = np.exp(C+D*vi+1j*phi*x)
    if(dPdv):
        integral = ((np.exp(-1j*phi*np.log(K))*D*f)/(1j*phi)).real
    else:
        integral = ((np.exp(-1j*phi*np.log(K))*f)/(1j*phi)).real
    return integral



