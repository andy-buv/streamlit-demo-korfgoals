import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics
from scipy.optimize import minimize, fmin
from scipy.special import comb
from scipy.special import factorial


# Make a github hosted file
df = pd.read_csv('https://raw.githubusercontent.com/andy-buv/streamlit-demo-korfgoals/master/eka_goals.csv')

goals = df['Goals']

x_max = goals.max() + 1
x = np.arange(0, x_max, 1)
x_arr = np.arange(0, x_max+1, 1)
y = np.histogram(goals, bins=x_arr, density=True)[0]

distributions = ("Poisson", "Normal", "Negative Binomial", "Gamma", "Binomial", "Geometric")

# x_arr = np.arange(0, 49, 1)

def get_dist_data(dist_name, params):
    if dist_name == "Poisson":
        dist = stats.poisson(params['Lambda']).pmf(x_arr)
    elif dist_name == "Normal":
        dist = stats.norm(params['Mu'], params['Std']).pdf(x_arr)
    elif dist_name == "Negative Binomial": 
        dist = stats.nbinom(params['N'], params['P']).pmf(x_arr)
    elif dist_name == "Gamma":
        dist = stats.gamma(params['A'], params['loc'], params['scale']).pdf(x_arr) 
    elif dist_name == "Binomial":
        dist = stats.binom(params['n'], params['p']).pmf(x_arr)
    elif dist_name == "Geometric":
        dist = stats.geom(params['p'],  loc=params['loc']).pmf(x_arr)
    return dist


def plot_data(data, dist, dist_name, params):
    f = plt.hist(data, bins=x_arr,
             edgecolor='white', alpha=0.5, density=True)
    plt.vlines(x_arr, 0, dist)
    plt.xlabel('Goals Scored')
    plt.ylabel('Relative Frequency')
    plt.annotate("(n = {0:,.0f} observations)".format(data.size), (.7, .9), 
                 xycoords='axes fraction', size=8)
    plt.title(f'{dist_name} Estimation of EKA Goals')
    st.pyplot()
    return f
    
    
def MLERegressionNBinom(params):
    n, p, sd = params[0], params[1], params[2]
    yhat =  comb(x+n-1, x)* (p**n) * ((1-p)**x)
    
    negLL = - np.sum(stats.norm.logpdf(y, loc=yhat, scale=sd))
    return negLL


def FitNBinom(data, guess):
    results = minimize(MLERegressionNBinom, guess, 
                       method='Nelder-Mead', options={'disp': True})
    if results.success:
        return results.x[:2]
    else:
        return 'Best Fit Not Found'
    
    
def MLERegressionPoisson(params):
    
    mu, sd = params[0], params[1]
    yhat = np.exp(-mu)*(mu**x)/factorial(x)
    
    negLL = -np.sum(stats.norm.logpdf(y, loc=yhat, scale=sd))
    return negLL


def FitPoisson(data, guess):
    
    results = minimize(MLERegressionPoisson, guess,
                       method='Nelder-Mead', options={'disp':True})
    
    if results.success:
        return results.x[0]
    else:
        return 'Best Fit Not Found'

def MLERegressionBinomial(params):
    n, p, sd = params[0], params[1], params[2]
    yhat = comb(n, x)*(p**x)*((1-p)**(n-x))
    
    negLL = -np.sum(stats.norm.logpdf(y, loc=yhat, scale=sd))
    return negLL

def FitBinomial(data, guess):
    
    results = minimize(MLERegressionBinomial, guess,
                       method='Nelder-Mead', options={'disp':True})
    print(results)
    if results.success:
        
        return results.x[:2]
    else:
        return results.x[:2]
    
def get_best_params(dist_name, data):
    if dist_name == "Poisson":
        guess = [20, 1]
        mu = FitPoisson(data, guess)
        return {'mu': mu}
    elif dist_name == "Normal":
        mu, std = stats.norm.fit(data)
        return {'mu': mu, 'std': std}
    elif dist_name == "Negative Binomial":
        guess = [12, .5, 1]
        N, p = FitNBinom(data, guess)
        return {'N': N, 'p': p}
    elif dist_name == 'Gamma':
        a, loc, scale = stats.gamma.fit(data)
        return {'a': a, 'loc': loc, 'scale': scale}
    elif dist_name == 'Binomial':
        guess = [10, .4, 1]
        n, p = FitBinomial(data, guess)
        return {'n': int(n), 'p': p}
    
    
def add_parameter_ui(dist_name):
    params = dict()
    best_params = get_best_params(dist_name, goals)
    if dist_name == "Poisson":
        l = st.sidebar.slider("Lambda", 0.00, 50.0, best_params['mu'])
        params['Lambda'] = l
        
    elif dist_name == "Normal":
        mu = st.sidebar.slider("Mu", 0.0, 50.0, best_params['mu'])
        std = st.sidebar.slider("Std", 0.0, 10.0, best_params['std'])
        params['Mu'] = mu
        params['Std'] = std
        
    elif dist_name == "Negative Binomial":
        n = st.sidebar.slider('N', 0.0, 50.0, best_params['N'])
        p = st.sidebar.slider('P', 0.010, 1.0, best_params['p'])
        params['N'] = n
        params['P'] = p
        
    elif dist_name == "Gamma":
        a = st.sidebar.slider('A', 0.0, 50.0, best_params['a'])
        loc = st.sidebar.slider('loc', -50., 50., best_params['loc'])
        scale = st.sidebar.slider('scale', 0.0, 20., best_params['scale'])
        params['A'] = a
        params['loc'] = loc
        params['scale'] = scale
    
    elif dist_name == "Binomial":
        n = st.sidebar.slider('n', 0, 100)
        p = st.sidebar.slider('p', 0.0, 1.0)
        params['n'] = n
        params['p'] = p
        
    elif dist_name == "Geometric":
        loc = st.sidebar.slider('loc', 0, 50)
        p = st.sidebar.slider('p', 0.0, 1.0)
        params['loc'] = loc
        params['p'] = p
    return params

    
def sse(actual, predicted):
    squared_errors = (actual - predicted)**2
    return np.sum(squared_errors)


def sst(actual):
    avg = np.mean(actual)
    squared_errors = (actual - avg) ** 2
    return np.sum(squared_errors)    


def main():
    st.write("""
    # Streamlit: EKA Goals Modelling App

    Statistical Estimations of Goals Scored in the England Korfball League Since 2014/15 Season""")
    
    

    st.write(f"""

    **Mean: ** {goals.mean():.2f}
    **Var: ** {goals.var():.2f}
    **Std: ** {goals.std():.2f}
    **Skew: ** {goals.skew():.2f}
    **Kurtosis: ** {goals.kurtosis():.2f}""")

    st.sidebar.header('Distribution Parameters')

    

    dist_name = st.sidebar.selectbox("Select Distribution", distributions)
    
    params = add_parameter_ui(dist_name)        

    dist = get_dist_data(dist_name, params)

    graph = plot_data(goals, dist, dist_name, params)

    g_data, dist_data = graph[0], dist[:-1]




    mse = metrics.mean_squared_error(g_data, dist_data)
    rmse = mse ** .5
    e_var = metrics.explained_variance_score(g_data, dist_data)
    max_error = metrics.max_error(g_data, dist_data)
    mae = metrics.mean_absolute_error(g_data, dist_data)
    # r2_score = metrics.r2_score(g_data, dist_data)
    sum_square_errors = sse(g_data, dist_data)
    total_sum_square = sst(g_data)


    st.write(f"""
    **MSE: ** {mse * 100 :.5f}
    **RMSE: ** {rmse : .5f}
    
    **Max Error:** {max_error: .5f}
    **MAE: ** {mae: .5f}
    **SSE: ** {sum_square_errors: .5f}
    **SST: ** {total_sum_square: .5f}

    **Explained Variance: ** {e_var:.3f}""")


if __name__ == '__main__':
    main()