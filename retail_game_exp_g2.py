import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from tqdm import tqdm
import sympy as sym
from scipy.special import lambertw
from scipy.integrate import quad
from scipy.special import factorial
from scipy.optimize import fsolve

#################################################
################ INPUTS #########################
#################################################

years = [2023, 2024]
fol_prices = "C:/Users/jujua/Desktop/Data_FR/prices_day_ahead_entsoe/"
fol_consos = "C:/Users/jujua/Desktop/Data_FR/profils_Enedis_RES/"

dic_prices = {}
for y in years:
    df = pd.read_csv(fol_prices+str(y)+".csv")
    dic_prices[y] = df['Prices'].to_numpy()

dic_prof_base = {}

for y in years:
    df = pd.read_excel(fol_consos+str(y)+".xlsx")
    df['dates'] = pd.to_datetime(df['HORODATE'])
    df = df.sort_values(by='dates').reset_index().drop(columns=['index','CATEGORIE','INDIC_DISPERS_POIDS_DYN','INDIC_PRECISION_DYN','HORODATE','SOUS_PROFIL'])
    #df = df.iloc[::2].reset_index().drop(columns=['index'])
    good_index = []
    for i in range(len(df['dates'])):
        if df['dates'][i].minute==0:
            good_index.append(i)
    df = df.iloc[good_index].reset_index().drop(columns=['index'])
    dic_prof_base[y] = df['COEFFICIENT_DYNAMIQUE'].to_numpy()

dic_prof_base['full'] = np.array(list(dic_prof_base[2023])+list(dic_prof_base[2024]))
dic_prices['full'] = np.array(list(dic_prices[2023])+list(dic_prices[2024]))[:-73]


def quant_weighted_avg_price(M,c,Q):    
    return np.sum(Q*c)/np.sum(Q)

def price_avg_per_time_block(M,c,Q):
    nb_block = int(len(c)/M)
    avgs = np.zeros(nb_block)
    Qsm = np.zeros(nb_block)
    for m in range(nb_block):
        Qsm[m] = np.sum(Q[m*M:(m+1)*M])
        avgs[m] = np.sum(Q[m*M:(m+1)*M]*c[m*M:(m+1)*M])/Qsm[m]
    return avgs, Qsm

def price_variance_per_block(M,c,Q,cm):
    nb_block = int(len(c)/M)
    var = 0
    for m in range(nb_block):
        for t in range(M):
            var += Q[t+m*M] * (c[t+m*M]**2-cm[m]**2)
    return var/np.sum(Q)

M=24
print("Time window for load shifting : %s h"%M)
Q = dic_prof_base['full']
Qtot = np.sum(Q)
print("Q total : %.2f kWh"%Qtot)
c = dic_prices['full'] / 10
cb = quant_weighted_avg_price(M,c,Q)
print("cb : %.2f c€/kWh"%cb)
cms, Qms = price_avg_per_time_block(M,c,Q)
variance = price_variance_per_block(M,c,Q,cms)
print("var_c : %.2f c€^2/kWh"%variance)

plot_init=False
if plot_init:
    fig,ax1=plt.subplots()
    ax1.plot(Qms,color='black')
    ax1.set_ylabel("Load averages per %s h time block"%M)
    ax1.set_xlabel("Days")
    ax2=ax1.twinx()
    ax2.plot(cms,color='red',linestyle="dashed")
    ax2.set_ylabel("Spot prices avgs (c€/kWh)",color='red')
    plt.show()

########################################################################################
#################### Dyn. retailer's best response #####################################
########################################################################################

########## BR dyn tariff - exponential ###########################
X = sym.symbols('X')    

def alpha(_nu,_lamb):
    return (_nu+_lamb)/(_nu/2+_lamb)

def valid_nu2(_cb,_varc,_lamb,_pi):
    poly = X**3 + (3*_lamb-2*(_cb-_pi)/_varc)*X**2 + _lamb*(_lamb-8*(_cb-_pi)/_varc)*X -_lamb**2 *(_lamb+8*(_cb-_pi)/_varc)
    roots = sym.solve(poly,X)
    good_roots = []
    for r in roots:
        if (sym.im(r) < 1e-10 and sym.im(r) > -1e-10):
            good_roots.append(float(sym.re(r).evalf()))
    return good_roots

def threshs_pi(_pi,_varc,_lamb,_beta,_cb):
    poly = X**3 + (3*_lamb-2*(_cb-_pi)/_varc)*X**2 + _lamb*(_lamb-8*(_cb-_pi)/_varc)*X -_lamb**2 *(_lamb+8*(_cb-_pi)/_varc)
    roots = sym.solve(poly,X)
    good_roots=[]
    for r in roots:
        if (sym.im(r) < 1e-10 and sym.im(r) > -1e-10):
            good_roots.append(float(sym.re(r).evalf()))
    values_thresh = []
    for r in good_roots:
        if r>=0:
            values_thresh.append(_varc/2 *(r+_lamb)/(r/2+_lamb) * ( (r+_lamb)* ((r+_lamb)/(r+2*_lamb) -1)- r/4))
    values_thresh.append(_cb-_beta*_lamb*_varc/4)
    return values_thresh

def pb_nu(_cb,_varc,_lamb,_pi):
    nus = valid_nu2(_cb,_varc,_lamb,_pi)
    pbs = []
    for nu in nus:
        pbs.append(_pi+_varc/8 * nu * alpha(nu,_lamb)**2)
    return pbs, nus


def Gd_moins(_Q,_cb,_varc,_lamb,_pi,_beta):
    return (_pi-_cb)*_Q + _beta*_lamb/4*_varc*_Q, 0, _pi, 1/2, _beta


def Gd_plus(_Q,_cb,_varc,_lamb,_pi,_beta):
    pbs,nus = pb_nu(_cb,_varc,_lamb,_pi)
    alphs = alpha(np.array(nus),_lamb)
    Gds=[]
    rates = []
    adops = []
    for i in range(len(nus)):
        a = _Q*_beta*np.exp(-nus[i]/_lamb)*(pbs[i]-_cb)
        b = _beta*(1-alphs[i]/2)*alphs[i]/2 * _varc *(_lamb+nus[i])*np.exp(-nus[i]/_lamb)*_Q
        Gds.append(a+b)
        rates.append(alpha(nus[i],_lamb)/2)
        adops.append(_beta*np.exp(-nus[i]/_lamb))
    return Gds, nus, pbs, rates, adops
    
    
def Gd(_Q,_cb,_varc,_lamb,_pi,_beta):
    gds, nu_m, pb_m, rate_m, adop_m = Gd_moins(_Q,_cb,_varc,_lamb,_pi,_beta)
    gplus,nuplus,pbplus, rate_plus, adop_plus = Gd_plus(_Q,_cb,_varc,_lamb,_pi,_beta)
    Gds = [gplus[i]*(nuplus[i]>0) for i in range(len(gplus))]+[gds]
    ind_max = np.argmax(Gds)
    threshs = threshs_pi(_pi,_varc,_lamb,_beta,_cb)
    if _pi<=np.min(threshs) or Gds[ind_max]<0:
        return 0, 2*(_cb-_pi)/_varc, _cb, 1 ,_beta*np.exp(-2*(_cb-_pi)/_lamb)
    else:
        if ind_max==len(Gds)-1 :
            return gds, nu_m, pb_m, rate_m, adop_m
        else:
            return gplus[ind_max],nuplus[ind_max],pbplus[ind_max], rate_plus[ind_max], adop_plus[ind_max]

# def rate(_nus,_pis,_cb,_varc,_lambs):
#     res=np.zeros(len(_nus))
#     for i in range(len(_nus)):
#         res[i] = 1 if _nus[i]==2*(_cb-_pis[i])/_varc else 0.5*(_lambs[i]+_nus[i])/_lambs[i]
#     return res

# def rate_unit(_nu,_pi,_cb,_varc,_lamb):
#     return 1 if _nu==2*(_cb-_pi)/_varc else 0.5*(_lamb+_nu)/_lamb 

def dyn_tariff_profile(_pb,_rate,_M,_c,_cb,_cms,_Qms):
    # choix arbitraire du prix de référence a cm + (pb-cb)/M * Q/Qm
    # dans la logique de cost reflection
    nb_block = int(len(_c)/_M)
    profile = np.zeros(len(_c))
    for d in range(nb_block):
        for t in range(_M):
            profile[t+_M*d] = _rate* (_c[t+_M*d]-_cms[d]) + _cms[d] + (_pb-_cb)/_M * np.sum(_Qms)/_Qms[d]
    return profile

def aggregate_load(_M,_Q,profile, dyn_avg_period, _nu,_lamb,_beta):
    nb_block = int(len(profile)/_M)
    res = np.zeros(len(profile))
    for d in range(nb_block):
        for t in range(_M):
            res[t+_M*d] = _Q[t+_M*d]* (1 - _beta*(_nu+_lamb)*np.exp(-_nu/_lamb)* (profile[t+24*d] - dyn_avg_period[d])   )
    return res

def full_RTP_load(_M,_c,_cms,_lamb,_beta):
    nb_block = int(len(_c)/_M)
    res = np.zeros(len(_c))    
    for d in range(nb_block):
        for t in range(24):
            res[t+24*d] = Q[t+24*d]* (1 - _beta*_lamb* (_c[t+24*d] - _cms[d])   )
    return res
# # Sensib tests
lamb=0.07285248 #FR fitted on HPHC
beta=0.8

plot_sensi_pi = True
if plot_sensi_pi:
    # BR sensitivity to pi at 2023-24 parameters        
    pis = np.arange(cb*0.8,cb*1.2,0.001)

    nus=[]
    pbs=[]
    Gds=[]
    rates = []
    adops = []
    for pi in tqdm(pis):
        aa,ab,ac,ar,ad = Gd(Qtot,cb,variance,lamb,pi,beta)
        Gds.append(aa)
        pbs.append(ac)
        nus.append(ab)
        rates.append(ar)
        adops.append(ad)

    ind_neg = np.where(np.array(Gds)<0)[0]
    pis_Gd_negs = pis[ind_neg]

    fig,axs=plt.subplots(2,2,figsize=(12,9))

    axs[0,0].plot(pis,adops,color="blue")
    axs[0,0].fill_between(pis_Gd_negs,0,0.8,color="gray",alpha=0.5)
    axs[0,0].set_xlabel(r'Flat tariff ($\pi$)')
    axs[0,0].set_ylabel("BR Adoption rates of dyn. tariff")
    axs[0,0].hlines(0.8,np.min(pis),np.max(pis),label='Prop. of flex. cons.',color='black',linestyle="dashed")
    axs[0,0].vlines(cb,0,0.8,label='Avg. cost',color='red',linestyle="dashed")
    axs[0,0].legend()

    axs[0,1].plot(pis,pbs,color="blue")
    axs[0,1].set_xlabel(r'Flat tariff ($\pi$)')
    axs[0,1].set_ylabel("BR dyn tariff's average (c€/kWh)")
    axs[0,1].plot(pis,pis,color='black',linestyle="dashed",label='Flat tariff')
    axs[0,1].fill_between(pis_Gd_negs,0,np.max(pis),color="gray",alpha=0.5)
    axs[0,1].vlines(cb,np.min(pis),np.max(pis),label='Avg. cost',color='red',linestyle="dashed")
    axs[0,1].legend()

    axs[1,0].plot(pis,np.array(Gds)/100,color="blue")
    axs[1,0].set_xlabel(r'Flat tariff ($\pi$)')
    axs[1,0].set_ylabel("BR profits of the dyn. retailer (€)")
    axs[1,0].vlines(cb,np.min(Gds)/100,np.max(Gds)/100,label='Avg. cost',color='red',linestyle="dashed")
    axs[1,0].hlines(0,np.min(pis),np.max(pis),color='black',linestyle="dashed",label="break even")
    axs[1,0].fill_between(pis_Gd_negs,np.min(Gds),np.max(Gds)/100,color="gray",alpha=0.5)
    axs[1,0].legend()

    axs[1,1].plot(pis,rates,color="blue") 
    axs[1,1].set_xlabel(r'Flat tariff ($\pi$)')
    axs[1,1].set_ylabel("BR dyn. tariff's rate-to-cost dynamics ratio")
    axs[1,1].hlines(1,np.min(pis),np.max(pis),color="black",linestyle="dashed",label="RTP's rate-to-cost dynamics")
    axs[1,1].fill_between(pis_Gd_negs,np.min(rates),np.max(rates),color="gray",alpha=0.5)
    axs[1,1].vlines(cb,0.8*np.min(rates),1.1*np.max(rates),label='Avg. cost',color='red',linestyle="dashed")
    axs[1,1].legend()

    plt.savefig("./Figures/BR_dyn_sensib_pi.pdf")
    plt.show()

pi=0.98*cb
plot_sensi_lamb_varc = True
if plot_sensi_lamb_varc:
    lambs = np.arange(0.01,5*lamb,0.001)
    nu_eqs1 = []
    pb_eqs1 = []
    G_eqs1 = []
    rate_eqs1 = []
    adop_eqs1 = []
    nu_eqs01 = []
    pb_eqs01 = []
    G_eqs01 = []
    rate_eqs01 = []
    adop_eqs01 = []
    nu_eqs10 = []
    pb_eqs10 = []
    G_eqs10 = []
    rate_eqs10 = []
    adop_eqs10 = []

    for l in tqdm(lambs):
        aa,ab,ac, ar, ad = Gd(Qtot,cb,variance,l,pi,beta)
        nu_eqs1.append(ab)
        pb_eqs1.append(ac)
        G_eqs1.append(aa)
        rate_eqs1.append(ar)
        adop_eqs1.append(ad)

        aa,ab,ac,ar,ad = Gd(Qtot,cb,variance/2,l,pi,beta)
        nu_eqs01.append(ab)
        pb_eqs01.append(ac)
        G_eqs01.append(aa)
        rate_eqs01.append(ar)
        adop_eqs01.append(ad)
        
        aa,ab,ac,ar,ad = Gd(Qtot,cb,variance*2,l,pi,beta)
        nu_eqs10.append(ab)
        pb_eqs10.append(ac)
        G_eqs10.append(aa)
        rate_eqs10.append(ar)
        adop_eqs10.append(ad)
    
    fig,axs=plt.subplots(2,2,figsize=(12,9))
    axs[0,0].plot(lambs,adop_eqs1,label="Var(c)",marker="o",color="blue",markevery=10,alpha=0.7)
    axs[0,0].plot(lambs,adop_eqs01,label="Var(c)/2",marker="+",color="green",markevery=10,alpha=0.4)
    axs[0,0].plot(lambs,adop_eqs10,label="2.varc",marker="v",color="purple",markevery=10,alpha=0.4)
    axs[0,0].vlines(lamb,0,np.max(adop_eqs10),color='black',linestyle='dashed',label='Fitted value FR')
    axs[0,0].set_xlabel(r'Avg. penalization $\lambda$')
    axs[0,0].set_ylabel("BR Adoption rates of dyn. tariff")
    axs[0,0].legend()
    axs[0,1].plot(lambs,pb_eqs1,label="Var(c)",marker="o",color="blue",markevery=10,alpha=0.7)
    axs[0,1].plot(lambs,pb_eqs01,label="Var(c)/2",marker="+",color="green",markevery=10,alpha=0.4)
    axs[0,1].plot(lambs,pb_eqs10,label="2.varc",marker="v",color="purple",markevery=10,alpha=0.4)
    axs[0,1].set_xlabel(r'Avg. penalization $\lambda$')
    axs[0,1].set_ylabel("BR dyn tariff's average")
    axs[0,1].hlines(pi,np.min(lambs),np.max(lambs),color='black',linestyle="dashed",label="Flat tariff")
    axs[0,1].hlines(cb,np.min(lambs),np.max(lambs),color='red',linestyle="dashed",label="Avg. cost")
    axs[0,1].legend()
    axs[1,0].plot(lambs,np.array(G_eqs1)/100,label="Var(c)",marker="o",color="blue",markevery=10,alpha=0.7)
    axs[1,0].plot(lambs,np.array(G_eqs01)/100,label="Var(c)/2",marker="+",color="green",markevery=10,alpha=0.4)
    axs[1,0].plot(lambs,np.array(G_eqs10)/100,label="2.varc",marker="v",color="purple",markevery=10,alpha=0.4)
    axs[1,0].set_xlabel(r'Avg. penalization $\lambda$')
    axs[1,0].set_ylabel("BR profits of the dyn. retailer (€)")
    axs[1,0].hlines(0,np.min(lambs),np.max(lambs),linestyle="dashed",color="black")
    axs[1,0].legend()
    axs[1,1].plot(lambs,rate_eqs1,label="Var(c)",marker="o",color="blue",markevery=10,alpha=0.7)
    axs[1,1].plot(lambs,rate_eqs01,label="Var(c)/2",marker="+",color="green",markevery=10,alpha=0.4)
    axs[1,1].plot(lambs,rate_eqs10,label="2.varc",marker="v",color="purple",markevery=10,alpha=0.4)
    axs[1,1].set_xlabel(r'Avg. penalization $\lambda$')
    axs[1,1].set_ylabel("BR dyn. tariff's rate-to-cost dynamics ratio")
    axs[1,1].hlines(1,np.min(lambs),np.max(lambs),color="black",linestyle="dashed",label="RTP's rate-to-cost dynamics")
    axs[1,1].legend()
    plt.savefig("./Figures/BR_dyn_sensib_lambda_varc.pdf")
    plt.show()

# # test one run
lamb=0.07285248 #FR fitted on HPHC
beta=0.8
pi = 0.98*cb
plot_one_test_br_dyn_exp = False
if plot_one_test_br_dyn_exp:
    ret_prof, br_nu, br_pb, br_rate, br_adop = Gd(Qtot,cb,variance,lamb,pi,beta)
    print("BR dyn tariff face au plat %s"%(pi))
    print("br_pb %s"%br_pb)
    print("br_nu %s"%br_nu)
    print("rate %s"%br_rate)
    print("adoption %s"%br_adop)
    br_profile = dyn_tariff_profile(br_pb,br_rate,M,c,cb,cms,Qms)
    avg_block_br_profile,_ = price_avg_per_time_block(M,br_profile,Q)
    new_load = aggregate_load(M,Q,br_profile,avg_block_br_profile,br_nu,lamb,beta)

    plt.figure()
    plt.plot(c,label='costs')
    plt.plot(br_profile,label="tariff",alpha=0.5)
    avgs=np.zeros(len(c))
    for d in range(int(len(c)/M)):
        for t in range(M):
            avgs[t+M*d] = avg_block_br_profile[d]
    plt.plot(avgs,label='avg tariff per block',color='red')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(Q,label='initial load')
    plt.plot(new_load,label="new load",alpha=0.5)
    plt.plot(full_RTP_load(M,c,cms,lamb,beta),label="full RTP load",alpha=0.5,color='green',linestyle='-.')
    plt.legend()
    plt.show()

########## BR dyn tariff - gamma 2 ###########################

def alpha_gl(x,t,k):
    num = 2*t+2*x+x**2 /t
    denom=2*t+1.5*x+0.5*x**2 /t
    return num/denom

def valid_nu_k2(_theta,_cb,_varc,_pi):
    X = sym.Symbol('X')
    nc = 2*(_cb-_pi)/_varc
    a = (1+X/_theta)*(2*_theta+2*X+X**2 / _theta)**2
    b = X/_theta**2 * (  X*(  4*( 2*_theta+1.5*X+0.5*X**2 /_theta)*(2*_theta+2*X+X**2 / _theta) - (2*_theta+2*X+X**2 / _theta)**2  ) + 4*nc* ( 2*_theta+1.5*X+0.5*X**2 /_theta)**2 )
    poly = sym.Poly(sym.simplify(a - b))
    roots = sym.solve(poly,X)
    good_roots = []
    for r in roots:
        if sym.im(r)<1e-6 and sym.im(r)>-1e-6 and r>=0.0:
            good_roots.append(float(r))
    return good_roots

def pb_nu_g2(_cb,_varc,_theta,_pi):
    nus = valid_nu_k2(_theta,_cb,_varc,_pi)
    pbs = []
    for nu in nus:
        pbs.append(_pi+_varc/8 * nu * alpha_gl(nu,_theta,2)**2)
    return pbs, nus

def Gd_moins_g2(_Q,_cb,_varc,_theta,_pi,_beta,_k=2):
    return (_pi-_cb)*_Q + _beta*_k*_theta/4*_varc*_Q, 0, _pi,1/2, _beta

def Gd_plus_g2(_Q,_cb,_varc,_theta,_pi,_beta,_k=2):
    pbs,nus = pb_nu_g2(_cb,_varc,_theta,_pi)
    if len(nus)==0:
        return 0, 2*(_cb-_pi)/_varc, _cb, 1, _beta*np.exp(-2*(_cb-_pi)/(_varc*_theta))*(1+2*(_cb-_pi)/(_varc*_theta))
    else:
        alphs = [alpha_gl(nu,_theta,_k) for nu in nus]
        Gds=[]
        rates=[]
        adops=[]
        for i in range(len(nus)):
            a = _Q*_beta*(pbs[i]-_cb)*np.exp(-nus[i]/_theta)*(1+nus[i]/_theta)
            b = _beta*(1-alphs[i]/2)*alphs[i]/2 * _varc *_Q *np.exp(-nus[i]/_theta)*(2*_theta+2*nus[i]+nus[i]**2 /_theta)
            Gds.append(a+b)
            rates.append(alpha_gl(nus[i],_theta,_k)/2)
            adops.append(_beta*np.exp(-nus[i]/_theta)*(1+nus[i]/_theta))
        ind_max = np.argmax(Gds)
        return Gds[ind_max], nus[ind_max], pbs[ind_max], rates[ind_max], adops[ind_max]

def Gd_g2(_Q,_cb,_varc,_theta,_pi,_beta,_k=2):
    gd_m, nu_m, pb_m,rate_m,adop_m = Gd_moins_g2(_Q,_cb,_varc,_theta,_pi,_beta,_k)
    gplus,nuplus,pbplus, rate_plus, adop_plus = Gd_plus_g2(_Q,_cb,_varc,_theta,_pi,_beta,_k)
    profits = [gplus,gd_m]
    ind_max = np.argmax(profits)
    if profits[ind_max]<0:
        return 0, 2*(_cb-_pi)/_varc, _cb,1,_beta*np.exp(-2*(_cb-_pi)/(_varc*_theta))*(1+2*(_cb-_pi)/(_varc*_theta))
    else:
        if ind_max==1 :
            return gd_m, nu_m, pb_m, rate_m, adop_m
        else:
            return gplus,nuplus,pbplus,rate_plus,adop_plus

def dyn_tariff_profile_g2(_pb,_rate,_M,_c,_cb,_cms,_Qms,_k=2):
    # choix arbitraire du prix de référence a cm + (pb-cb)/M * Q/Qm
    # dans la logique de cost reflection
    nb_block = int(len(_c)/_M)
    profile = np.zeros(len(_c))
    for d in range(nb_block):
        for t in range(_M):
            profile[t+_M*d] = _rate* (_c[t+_M*d]-_cms[d]) + _cms[d] + (_pb-_cb)/_M * np.sum(_Qms)/_Qms[d]
    return profile

def aggregate_load_g2(_M,_Q,profile, dyn_avg_period, _nu,_theta,_beta,_k=2):
    nb_block = int(len(profile)/_M)
    res = np.zeros(len(profile))
    for d in range(nb_block):
        for t in range(_M):
            res[t+_M*d] = _Q[t+_M*d]* (1 - _beta*_k*_theta*np.exp(-_nu/_theta)*np.sum([_nu**l / (factorial(l)*_theta**l) for l in range(_k+1)])* (profile[t+_M*d] - dyn_avg_period[d])   )
    return res

def full_RTP_load_g2(_M,_Q,_c,_cms,_theta,_beta,_k=2):
    res = np.zeros(len(_c))
    for d in range(int(len(_c)/_M)):
        for t in range(_M):
            res[t+_M*d] = _Q[t+_M*d]* (1 - _beta*_k*_theta* (_c[t+_M*d] - _cms[d])   )
    return res

# # test one run
plot_one_test_br_dyn_g2 = False
theta=0.03750787 #FR fitted on HPHC vs Base for beta=0.8
k=2
beta=0.8
pi = 0.98*cb
if plot_one_test_br_dyn_g2:
    ret_prof, br_nu, br_pb,br_rate,br_adop = Gd_g2(Qtot,cb,variance,lamb,pi,beta)
    print("BR dyn tariff face au plat %s"%(pi))
    print("br_pb %s"%br_pb)
    print("br_nu %s"%br_nu)
    print("rate %s"%br_rate)
    print("adoption %s"%br_adop)
    br_profile = dyn_tariff_profile_g2(br_pb, br_rate, M, c, cb, cms, Qms)
    avg_block_br_profile,_ = price_avg_per_time_block(M,br_profile,Q)
    new_load = aggregate_load_g2(M,Q,br_profile,avg_block_br_profile,br_nu,theta,beta)

    plt.figure()
    plt.plot(c,label='costs')
    plt.plot(br_profile,label="tariff",alpha=0.7)
    avgs=np.zeros(len(c))
    for d in range(int(len(c)/M)):
        for t in range(M):
            avgs[t+M*d] = avg_block_br_profile[d]
    plt.plot(avgs,label='avg tariff per block',color='red')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(Q,label='initial load')
    plt.plot(new_load,label="new load",alpha=0.2)
    plt.plot(full_RTP_load_g2(M,Q,c,cms,theta,beta),label="full RTP load",alpha=0.5,color='green',linestyle='-.')
    plt.legend()
    plt.show()

plot_sensi_pi_g2 = True
if plot_sensi_pi_g2:
    pis = np.arange(cb*0.8,cb*1.2,0.001)
    nus = []
    pbs = []
    Gds = []
    rates = []
    adops = []
    for pi in tqdm(pis):
        aa,ab,ac,ar,ad = Gd_g2(Qtot,cb,variance,theta,pi,beta)
        Gds.append(aa)
        pbs.append(ac)
        nus.append(ab)
        rates.append(ar)
        adops.append(ad)
    ind_neg = np.where(np.array(Gds)<0)[0]
    pis_Gd_negs = pis[ind_neg]

    fig,axs=plt.subplots(2,2,figsize=(12,9))

    axs[0,0].plot(pis,adops,color="blue")
    axs[0,0].fill_between(pis_Gd_negs,0,0.8,color="gray",alpha=0.5)
    axs[0,0].set_xlabel(r'Flat tariff ($\pi$)')
    axs[0,0].set_ylabel("BR Adoption rates of dyn. tariff")
    axs[0,0].hlines(beta,np.min(pis),np.max(pis),label='Prop. of flex. cons.',color='black',linestyle="dashed")
    axs[0,0].vlines(cb,0,0.8,label='Avg. cost',color='red',linestyle="dashed")
    axs[0,0].legend()

    axs[0,1].plot(pis,pbs,color="blue")
    axs[0,1].set_xlabel(r'Flat tariff ($\pi$)')
    axs[0,1].set_ylabel("BR dyn tariff's average (c€/kWh)")
    axs[0,1].plot(pis,pis,color='black',linestyle="dashed",label='Flat tariff')
    axs[0,1].fill_between(pis_Gd_negs,0,np.max(pis),color="gray",alpha=0.5)
    axs[0,1].vlines(cb,np.min(pis),np.max(pis),label='Avg. cost',color='red',linestyle="dashed")
    axs[0,1].legend()

    axs[1,0].plot(pis,np.array(Gds)/100,color="blue")
    axs[1,0].set_xlabel(r'Flat tariff ($\pi$)')
    axs[1,0].set_ylabel("BR profits of the dyn. retailer (€)")
    axs[1,0].vlines(cb,np.min(Gds)/100,np.max(Gds)/100,label='Avg. cost',color='red',linestyle="dashed")
    axs[1,0].hlines(0,np.min(pis),np.max(pis),color='black',linestyle="dashed",label="break even")
    axs[1,0].fill_between(pis_Gd_negs,np.min(Gds)/100,np.max(Gds)/100,color="gray",alpha=0.5)
    axs[1,0].legend()

    axs[1,1].plot(pis,rates,color="blue") 
    axs[1,1].set_xlabel(r'Flat tariff ($\pi$)')
    axs[1,1].set_ylabel("BR dyn. tariff's rate-to-cost dynamics ratio")
    axs[1,1].hlines(1,np.min(pis),np.max(pis),color="black",linestyle="dashed",label="RTP's rate-to-cost dynamics")
    axs[1,1].fill_between(pis_Gd_negs,np.min(rates),np.max(rates),color="gray",alpha=0.5)
    axs[1,1].vlines(cb,0.8*np.min(rates),1.1*np.max(rates),label='Avg. cost',color='red',linestyle="dashed")
    axs[1,1].legend()

    plt.savefig("./Figures/BR_dyn_sensib_pi_g2.pdf")
    plt.show()

plot_sensi_theta_varc_g2 = True
if plot_sensi_theta_varc_g2:
    thetas = np.arange(0.01,10*theta,0.001)
    pi=0.98*cb
    nu_eqs1 = []
    pb_eqs1 = []
    G_eqs1 = []
    rate_eqs1 = []
    adop_eqs1 = []
    nu_eqs01 = []
    pb_eqs01 = []
    G_eqs01 = []
    rate_eqs01 = []
    adop_eqs01 = []
    nu_eqs10 = []
    pb_eqs10 = []
    G_eqs10 = []
    rate_eqs10 = []
    adop_eqs10 = []

    for l in tqdm(thetas):
        aa,ab,ac,ar,ad = Gd_g2(Qtot,cb,variance,l,pi,beta)
        nu_eqs1.append(ab)
        pb_eqs1.append(ac)
        G_eqs1.append(aa)
        rate_eqs1.append(ar)
        adop_eqs1.append(ad)

        aa,ab,ac,ar,ad = Gd_g2(Qtot,cb,variance/2,l,pi,beta)
        nu_eqs01.append(ab)
        pb_eqs01.append(ac)
        G_eqs01.append(aa)
        rate_eqs01.append(ar)
        adop_eqs01.append(ad)
        
        aa,ab,ac,ar,ad = Gd_g2(Qtot,cb,2*variance,l,pi,beta)
        nu_eqs10.append(ab)
        pb_eqs10.append(ac)
        G_eqs10.append(aa)
        rate_eqs10.append(ar)
        adop_eqs10.append(ad)

    fig,axs=plt.subplots(2,2,figsize=(12,9))
    axs[0,0].plot(2*thetas,adop_eqs1,label="Var(c)",marker="o",color="blue",markevery=10,alpha=0.7)
    axs[0,0].plot(2*thetas,adop_eqs01,label="Var(c)/2",marker="+",color="green",markevery=10,alpha=0.4)
    axs[0,0].plot(2*thetas,adop_eqs10,label="2.Var(c)",marker="v",color="purple",markevery=10,alpha=0.4)
    axs[0,0].vlines(2*theta,0,np.max(adop_eqs10),color='black',linestyle='dashed',label='Fitted value FR')
    axs[0,0].set_xlabel(r'Avg. penalization $2\theta$')
    axs[0,0].set_ylabel("BR Adoption rates of dyn. tariff")
    axs[0,0].legend()
    axs[0,1].plot(2*thetas,pb_eqs1,label="Var(c)",marker="o",color="blue",markevery=10,alpha=0.7)
    axs[0,1].plot(2*thetas,pb_eqs01,label="Var(c)/2",marker="+",color="green",markevery=10,alpha=0.4)
    axs[0,1].plot(2*thetas,pb_eqs10,label="2.Var(c)",marker="v",color="purple",markevery=10,alpha=0.4)
    axs[0,1].set_xlabel(r'Avg. penalization $2\theta$')
    axs[0,1].set_ylabel("BR dyn tariff's average (c€/kWh)")
    axs[0,1].hlines(pi,np.min(thetas),2*np.max(thetas),color='black',linestyle="dashed",label="Flat tariff")
    axs[0,1].hlines(cb,np.min(thetas),2*np.max(thetas),color='red',linestyle="dashed",label="Avg. cost")
    axs[0,1].legend()
    axs[1,0].plot(2*thetas,np.array(G_eqs1)/100,label="Var(c)",marker="o",color="blue",markevery=10,alpha=0.7)
    axs[1,0].plot(2*thetas,np.array(G_eqs01)/100,label="Var(c)/2",marker="+",color="green",markevery=10,alpha=0.4)
    axs[1,0].plot(2*thetas,np.array(G_eqs10)/100,label="2.Var(c)",marker="v",color="purple",markevery=10,alpha=0.4)
    axs[1,0].set_xlabel(r'Avg. penalization $2\theta$')
    axs[1,0].set_ylabel("BR profits of the dyn. retailer (€)")
    axs[1,0].hlines(0,np.min(thetas),2*np.max(thetas),linestyle="dashed",color="black")
    axs[1,0].legend()
    axs[1,1].plot(2*thetas,rate_eqs1,label="Var(c)",marker="o",color="blue",markevery=10,alpha=0.7)
    axs[1,1].plot(2*thetas,rate_eqs01,label="Var(c)/2",marker="+",color="green",markevery=10,alpha=0.4)
    axs[1,1].plot(2*thetas,rate_eqs10,label="2.Var(c)",marker="v",color="purple",markevery=10,alpha=0.4)
    axs[1,1].set_xlabel(r'Avg. penalization $2\theta$')
    axs[1,1].set_ylabel("BR dyn. tariff's rate-to-cost dynamics ratio")
    axs[1,1].hlines(1,np.min(thetas),2*np.max(thetas),color="black",linestyle="dashed",label="RTP's rate-to-cost dynamics")
    axs[1,1].legend()
    plt.savefig("./Figures/BR_dyn_sensib_lambda_varc_g2.pdf")
    plt.show()
    
########################################################################################
#################### Flat retailer's best response #####################################
########################################################################################

########### BR flat tariff - exponential ##########################

def br_pi(_pb,_varp,_lamb,_cb0,_beta):
    mid_pi = _cb0 - _lamb*_varp/2 + _lamb*_varp/2 * np.real(lambertw(1/_beta * np.exp(1+2*(_pb-_cb0)/(_lamb*_varp))))
    pi_res = np.max([_cb0,np.min([_pb,mid_pi])])
    nu = 2*(_pb-pi_res)/_varp
    return pi_res, nu

def Gf(_Q,_pb,_varp,_lamb,_cb0,_beta):
    pi, nu = br_pi(_pb,_varp,_lamb,_cb0,_beta)
    if nu>=0:
        if pi > _cb0:
            return (pi-_cb0)*_Q*(1-_beta+_beta*(1-np.exp(-nu/_lamb))),nu,pi, _beta*np.exp(-nu/_lamb)
        else:
            nu_base = 2*(_pb-_cb0)/_varp
            adop_base = 1 if nu_base < 0 else _beta*np.exp(-nu_base/_lamb)
            return 0, nu_base, _cb0, adop_base
    else:
        if _pb > _cb0:
            return (_pb-_cb0)*_Q*(1-_beta),0,_pb, _beta
        else:
            nu_base = 2*(_pb-_cb0)/_varp
            adop_base = 1 if nu_base < 0 else _beta*np.exp(-nu_base/_lamb)
            return 0, nu_base, _cb0, adop_base


# Test one run
test_one_run_flat_exp = False
if test_one_run_flat_exp:
    pb=cb
    cb0=0.98*cb
    beta=0.8
    lamb =0.07285248 #FR fitted on HPHC
    prof_flat, br_nu_flat, br_pi_flat, br_adop_dyn = Gf(Qtot,pb,variance,lamb,cb0,beta)
    print("BR flat - exponential facing pb = %.2f; var(p) = %.2f"%(pb,variance))
    print("br_pi %.2f"%br_pi_flat)
    print("br_nu_flat %.2f"%br_nu_flat)
    print("adoption %s"%(1-br_adop_dyn))

# Sensitivity to pb and var(p)
plot_sensi_pb = True
if plot_sensi_pb:
    cb0 = 0.98*cb
    pbs = np.arange(cb0-1,cb0+1,0.001)
    lamb =0.07285248 #FR fitted on HPHC
    beta=0.8
    
    Gfs=[]
    Gfs01=[]
    Gfs10=[]
    nus=[]
    nus01=[]
    nus10=[]
    adops=[]
    adops01=[]
    adops10=[]
    pis=[]
    pis01=[]
    pis10=[]
    for pb in tqdm(pbs):
        gf,n,pii,adop_dyn = Gf(Qtot,pb,variance,lamb,cb0,beta)
        Gfs.append(gf)
        nus.append(n)
        pis.append(pii)
        adops.append(1-adop_dyn)
        
        gf,n,pii,adop_dyn = Gf(Qtot,pb,variance/2,lamb,cb0,beta)
        Gfs01.append(gf)
        nus01.append(n)
        pis01.append(pii)
        adops01.append(1-adop_dyn)
        
        gf,n,pii,adop_dyn = Gf(Qtot,pb,variance*2,lamb,cb0,beta)
        Gfs10.append(gf)
        nus10.append(n)
        pis10.append(pii)
        adops10.append(1-adop_dyn)
    
    
    fig,axs=plt.subplots(1,3,figsize=(18,4))

    axs[0].plot(pbs,adops,label="Var(p)=Var(c)",marker="o",color="blue",markevery=100,alpha=0.7)
    axs[0].plot(pbs,adops01,label="Var(p)=Var(c)/2",marker="+",color="green",markevery=100,alpha=0.4)
    axs[0].plot(pbs,adops10,label="Var(p)=Var(c)*2",marker="v",color="purple",markevery=100,alpha=0.4)
    axs[0].hlines(1,np.min(pbs),np.max(pbs),color="black",linestyle="dashed")
    axs[0].hlines(beta,np.min(pbs),np.max(pbs),color="red",linestyle="dashed",label="Prop. of flex. cons.")
    axs[0].vlines(cb0,0,1,color="black",linestyle="-.",label="Flat. ret. avg cost")
    axs[0].set_xlabel("Avg. dyn. tariff (c€/kWh)")
    axs[0].set_ylabel("BR Adoption rate of flat tariff")
    axs[0].legend()

    axs[1].plot(pbs,pis,label="Var(p)=Var(c)",marker="o",color="blue",markevery=100,alpha=0.7)
    axs[1].plot(pbs,pis01,label="Var(p)=Var(c)/2",marker="+",color="green",markevery=100,alpha=0.4)
    axs[1].plot(pbs,pis10,label="Var(p)=Var(c)*2",marker="v",color="purple",markevery=100,alpha=0.4)
    axs[1].plot(pbs,pbs,color='black',linestyle="dashed",label="Avg. dyn. tariff")
    axs[1].vlines(cb0,np.min(pis)*0.9,1.1*np.max(pis),color="black",linestyle="-.",label="Flat. ret. avg cost")
    axs[1].set_xlabel("Avg. dyn. tariff (c€/kWh)")
    axs[1].set_ylabel("BR flat tariff (c€/kWh)")
    axs[1].legend()

    axs[2].plot(pbs,np.array(Gfs)/100,label="Var(p)=Var(c)",marker="o",color="blue",markevery=100,alpha=0.7)
    axs[2].plot(pbs,np.array(Gfs01)/100,label="Var(p)=Var(c)/2",marker="+",color="green",markevery=100,alpha=0.4)
    axs[2].plot(pbs,np.array(Gfs10)/100,label="Var(p)=Var(c)*2",marker="v",color="purple",markevery=100,alpha=0.4)
    axs[2].set_xlabel("Avg. dyn. tariff (c€/kWh)")
    axs[2].set_ylabel("BR profits of the flat retailer (€)")
    axs[2].hlines(0,np.min(pbs),np.max(pbs),color='black',linestyle="dashed",label="break even")
    axs[2].vlines(cb0,np.min(Gfs01)*0.9/100,1.1*np.max(Gfs10)/100,color="black",linestyle="-.",label="Flat. ret. avg cost")
    axs[2].legend()
    plt.savefig("./Figures/BR_flat_sensib_pb.pdf")
    plt.show()


########### BR flat tarrif - gamma 2 ##############################

def br_pi_interior_g2(_pb,_varp,_cb0,_theta,_beta):
    n0 = 2*(_pb-_cb0)/_varp
    def f(x):
        return np.exp(x/_theta)/_beta + x**2 / _theta**2 - n0*x/_theta**2 - 1 - x/_theta
    xstar = _theta/2+n0/2 - _theta*np.real(lambertw( np.exp(0.5+0.5*n0/_theta) / (2*_beta) ))
    ystar = f(xstar)
    if ystar>0:
        return _pb,0
    else:
        root = fsolve(f,xstar+0.2)[0]
        return _pb - 0.5*_varp*root, root

def Gf_g2(_Q,_pb,_varp,_cb0,_theta,_beta,_k=2):
    pi, nu = br_pi_interior_g2(_pb,_varp,_cb0,_theta,_beta)
    if nu>=0:
        if pi > _cb0:
            return (pi-_cb0)*_Q*(1-_beta+_beta*(1-np.exp(-nu/_theta)*(1+nu/_theta)   )),nu,pi, _beta*np.exp(-nu/_theta)*(1+nu/_theta) 
        else:
            nu_base = 2*(_pb-_cb0)/_varp
            adop_base = 1 if nu_base < 0 else _beta*np.exp(-2*(_pb-_cb0)/(_varp*_theta))*(1+2*(_pb-_cb0)/(_varp*_theta)) 
            return 0, nu_base, _cb0, adop_base
    else:
        if _pb > _cb0:
            return (_pb-_cb0)*_Q*(1-_beta),0,_pb, _beta
        else:
            nu_base = 2*(_pb-_cb0)/_varp
            adop_base = 1 if nu_base < 0 else _beta*np.exp(-2*(_pb-_cb0)/(_varp*_theta))*(1+2*(_pb-_cb0)/(_varp*_theta)) 
            return 0, nu_base, _cb0, adop_base
# Test one run
test_one_run_flat_g2 = True
if test_one_run_flat_g2:
    pb=cb
    cb0=0.98*cb
    beta=0.8
    k=2
    theta=0.03750787 #FR fitted on HPHC vs Base for beta=0.8
    prof_flat, br_nu_flat, br_pi_flat, br_adop_dyn = Gf_g2(Qtot,pb,variance,cb0,theta,beta)
    print("BR flat - G2 facing pb = %.2f; var(p) = %.2f"%(pb,variance))
    print("br_pi %.2f"%br_pi_flat)
    print("br_nu_flat %.2f"%br_nu_flat)
    print("adoption %s"%(1-br_adop_dyn))

plot_sensi_pb_g2 = True
if plot_sensi_pb_g2:
    cb0 = 0.98*cb
    pbs = np.arange(cb0-1,cb0+1,0.001)
    theta=0.03750787 #FR fitted on HPHC vs Base for beta=0.8
    k=2
    beta=0.8

    Gfs=[]
    Gfs01=[]
    Gfs10=[]
    nus=[]
    nus01=[]
    nus10=[]
    adops=[]
    adops01=[]
    adops10=[]
    pis=[]
    pis01=[]
    pis10=[]
    for pb in tqdm(pbs):
        gf,n,pii,ad = Gf_g2(Qtot,pb,variance,cb0,theta,beta)
        Gfs.append(gf)
        nus.append(n)
        pis.append(pii)
        adops.append(1-ad)
        
        gf,n,pii,ad = Gf_g2(Qtot,pb,variance/2,cb0,theta,beta)
        Gfs01.append(gf)
        nus01.append(n)
        adops01.append(1-ad)
        pis01.append(pii)
        
        gf,n,pii,ad = Gf_g2(Qtot,pb,variance*2,cb0,theta,beta)
        Gfs10.append(gf)
        nus10.append(n)
        adops10.append(1-ad)
        pis10.append(pii)
        
    fig,axs=plt.subplots(1,3,figsize=(18,4))

    axs[0].plot(pbs,adops,label="Var(p)=Var(c)",marker="o",color="blue",markevery=100,alpha=0.7)
    axs[0].plot(pbs,adops01,label="Var(p)=Var(c)/2",marker="+",color="green",markevery=100,alpha=0.4)
    axs[0].plot(pbs,adops10,label="Var(p)=Var(c)*2",marker="v",color="purple",markevery=100,alpha=0.4)
    axs[0].hlines(1,np.min(pbs),np.max(pbs),color="black",linestyle="dashed")
    axs[0].hlines(beta,np.min(pbs),np.max(pbs),color="red",linestyle="dashed",label="Prop. of flex. cons.")
    axs[0].vlines(cb0,0,1,color="black",linestyle="-.",label="Flat. ret. avg cost")
    axs[0].set_xlabel("Avg. dyn. tariff (c€/kWh)")
    axs[0].set_ylabel("BR Adoption rate of flat tariff")
    axs[0].legend()

    axs[1].plot(pbs,pis,label="Var(p)=Var(c)",marker="o",color="blue",markevery=100,alpha=0.7)
    axs[1].plot(pbs,pis01,label="Var(p)=Var(c)/2",marker="+",color="green",markevery=100,alpha=0.4)
    axs[1].plot(pbs,pis10,label="Var(p)=Var(c)*2",marker="v",color="purple",markevery=100,alpha=0.4)
    axs[1].plot(pbs,pbs,color='black',linestyle="dashed",label="Avg. dyn. tariff")
    axs[1].vlines(cb0,np.min(pis)*0.9,1.1*np.max(pis),color="black",linestyle="-.",label="Flat. ret. avg cost")
    axs[1].set_xlabel("Avg. dyn. tariff (c€/kWh)")
    axs[1].set_ylabel("BR flat tariff (c€/kWh)")
    axs[1].legend()

    axs[2].plot(pbs,np.array(Gfs)/100,label="Var(p)=Var(c)",marker="o",color="blue",markevery=100,alpha=0.7)
    axs[2].plot(pbs,np.array(Gfs01)/100,label="Var(p)=Var(c)/2",marker="+",color="green",markevery=100,alpha=0.4)
    axs[2].plot(pbs,np.array(Gfs10)/100,label="Var(p)=Var(c)*2",marker="v",color="purple",markevery=100,alpha=0.4)
    axs[2].set_xlabel("Avg. dyn. tariff (c€/kWh)")
    axs[2].set_ylabel("BR profits of the flat retailer (€)")
    axs[2].hlines(0,np.min(pbs),np.max(pbs),color='black',linestyle="dashed",label="break even")
    axs[2].vlines(cb0,np.min(Gfs01)*0.9/100,1.1*np.max(Gfs10)/100,color="black",linestyle="-.",label="Flat. ret. avg cost")
    axs[2].legend()
    plt.savefig("./Figures/BR_flat_sensib_pb_g2.pdf")
    plt.show()
