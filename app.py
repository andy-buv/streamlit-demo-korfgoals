import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics
from scipy.optimize import minimize, fmin
from scipy.special import comb
from scipy.special import factorial


datasets = ('England Korfball League',)
distributions = ("Poisson", "Normal", "Negative Binomial", "Gamma", "Binomial")


data_source = {'England Korfball League': 'https://raw.githubusercontent.com/andy-buv/streamlit-demo-korfgoals/master/eka_league_data.csv'}

st.sidebar.header('Dataset Filters')
dataset = st.sidebar.selectbox("Select Dataset", datasets)

df = pd.read_csv(data_source[dataset])

seasons = st.sidebar.multiselect("Select Seasons", df.Season.unique())
teams = st.sidebar.multiselect("Select Teams", df['Home Team'].sort_values().unique())


def filter_scores(data, teams, seasons):
    if seasons != []:
        data = data.loc[data['Season'].isin(seasons)]
    
    if teams != []:
        home_scores = data.loc[data['Home Team'].isin(teams)]
        away_scores = data.loc[data['Away Team'].isin(teams)]
    else:
        home_scores = data
        away_scores = data
        
    stacked_scores = pd.concat([home_scores['Home Score'], away_scores['Away Score']], 
                               ignore_index=True)
    stacked_scores = pd.DataFrame(stacked_scores).dropna()
    stacked_scores.columns = ['Goals']
    
    return stacked_scores


goals = filter_scores(df, teams, seasons)['Goals']

st.sidebar.header('Histogram')

bin_size = st.sidebar.number_input('Bin Size', 1, 10, 2)
x_max = goals.max() + 1
x = np.arange(0, x_max, bin_size)
x_arr = np.arange(0, x_max + bin_size, bin_size)
y = np.histogram(goals, bins=x_arr, density=True)[0]


def get_dist_data(dist_name, params):
    if dist_name == "Poisson":
        dist = stats.poisson(params['mu']).pmf(x_arr)
    elif dist_name == "Normal":
        dist = stats.norm(params['loc'], params['scale']).pdf(x_arr)
    elif dist_name == "Negative Binomial": 
        dist = stats.nbinom(params['n'], params['p']).pmf(x_arr)
    elif dist_name == "Gamma":
        dist = stats.gamma(params['a'], params['loc'], params['scale']).pdf(x_arr) 
    elif dist_name == "Binomial":
        dist = stats.binom(params['n'], params['p']).pmf(x_arr)
    elif dist_name == "Log-Normal":
        dist = stats.lognorm.pdf(x_arr, params['s'],  loc=params['loc'], scale=params['scale'] )
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
    yhat = comb(x+n-1, x)* (p**n) * ((1-p)**x)
    
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
        loc, scale = stats.norm.fit(data)
        return {'loc': loc, 'scale': scale}
    elif dist_name == "Negative Binomial":
        guess = [12, .5, 1]
        n, p = FitNBinom(data, guess)
        return {'n': n, 'p': p}
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
        mu = st.sidebar.slider("μ", 0.00, 50.0, best_params['mu'])
        params['mu'] = mu
        
    elif dist_name == "Normal":
        loc = st.sidebar.slider("loc", 0.0, 50.0, best_params['loc'])
        scale = st.sidebar.slider("scale", 0.0, 10.0, best_params['scale'])
        params['loc'] = loc
        params['scale'] = scale
        
    elif dist_name == "Negative Binomial":
        n = st.sidebar.slider('n', 0.0, 50.0, best_params['n'])
        p = st.sidebar.slider('p', 0.010, 1.0, best_params['p'])
        params['n'] = n
        params['p'] = p
        
    elif dist_name == "Gamma":
        a = st.sidebar.slider('a', 0.0, 50.0, best_params['a'])
        loc = st.sidebar.slider('loc', -50., 50., best_params['loc'])
        scale = st.sidebar.slider('scale', 0.0, 20., best_params['scale'])
        params['a'] = a
        params['loc'] = loc
        params['scale'] = scale
    
    elif dist_name == "Binomial":
        n = st.sidebar.slider('n', 0, 3000, best_params['n'])
        p = st.sidebar.slider('p', 0.0, 1.0, best_params['p'])
        params['n'] = n
        params['p'] = p
        
    elif dist_name == "Log-Normal":
        s = st.sidebar.slider('s', 0.001, 50.0)
        loc = st.sidebar.slider('loc', 0, 50)
        scale = st.sidebar.slider('scale', 0.01, 10.0, 1.0)
        params['s'] = s
        params['loc'] = loc
        params['scale'] = scale

    return params

def get_formula(dist_name):
    
    formulas = {'Normal': r"""
The probability density function for norm is:

$f(x) = \frac{\exp(-x^2/2)}{\sqrt{2\pi}}$
    
for a real number $x$.""",
               'Poisson': r"""
The probabilty mass function for poisson is:

$f(k) = \exp(-\mu) \frac{\mu^k}{k!}$

for $k \ge 0$.
poisson takes $\mu$ as shape parameter.""",
               'Negative Binomial': r"""
The probability mass function of the number of failures for nbinom is:

$f(k) = \binom{k+n-1}{n-1} p^n (1-p)^k$

for $k \ge 0$.""",
               'Gamma': r"""
The probability density function for gamma is:

$f(x, a, b) = \frac{\Gamma(a+b) x^{a-1} (1-x)^{b-1}}
                          {\Gamma(a) \Gamma(b)}$               

for $x \ge 0, a \gt 0$. Here $\Gamma(a)$ refers to the gamma function.""",
               'Binomial': r"""
The probability mass function for binomial is:

$f(k) = \binom{n}{k} p^k (1-p)^{n-k}$

for $k$ in $(0, 1,..., n)$."""}
    
    return formulas[dist_name]
    
    
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

    st.sidebar.markdown('#### Formula')
    formula = st.sidebar.markdown(get_formula(dist_name))
    
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