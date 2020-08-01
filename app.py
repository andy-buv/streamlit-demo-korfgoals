import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics
from scipy.optimize import minimize, fmin
from scipy.special import comb
from scipy.special import factorial
import altair as alt


datasets = ('England Korfball League',)
distributions = ("Poisson", "Normal", "Negative Binomial", "Gamma", "Binomial")
venues = ("Home", "Away")

data_sources = {'England Korfball League': 'https://raw.githubusercontent.com/andy-buv/' \
                                          'streamlit-demo-korfgoals/master/eka_league_data.csv'}



def filter_scores(data, teams, seasons, venue):
    
    if seasons != []:
        data = data.loc[data['Season'].isin(seasons)]
    
    if teams != []:
        home_scores = data.loc[data['Home Team'].isin(teams)]
        away_scores = data.loc[data['Away Team'].isin(teams)]
    else:
        home_scores = data
        away_scores = data
    
    if venue == ['Home']:
        stacked_scores = home_scores['Home Score']
    elif venue == ['Away']:
        stacked_scores = away_scores['Away Score']
    else:    
        stacked_scores = pd.concat([home_scores['Home Score'], away_scores['Away Score']], 
                                   ignore_index=True)
        
    stacked_scores = pd.DataFrame(stacked_scores).dropna()
    stacked_scores.columns = ['Goals']
    
    return stacked_scores





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


### CONVERT PLOTTING CHART TO ALTAIR
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
    
def plot_altair(hist, dist, dist_name, bin_size):
    
    brush = alt.selection_interval(encodings=['x'])
    
    data = pd.DataFrame.from_dict({'rf': hist, 
                         'p': dist}, orient='index').transpose().fillna(0).reset_index()
    
    data['index'] = data['index'] * bin_size

    base = alt.Chart(data, title=f'{dist_name} Estimation of EKA Goals').encode(
        alt.X('index:Q', title='Goals Scored', 
              bin=alt.Bin(step=bin_size), axis=alt.Axis(values=[-5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])))
    
    bar = base.mark_bar(opacity=.7).encode(
        alt.Y('rf:Q'))
    
    rule = base.mark_rule(size=2).encode(alt.X('index:Q'),
        alt.Y('p:Q', title='Relative Frequency', axis=alt.Axis(tickCount=5)))
    
    
    return alt.layer(bar, rule).properties(width=600, height=500).configure_axis(titleFontSize=16).configure_title(fontSize=20)

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
        loc, scale = stats.norm.fit(goals)
        return {'loc': loc, 'scale': scale}
    elif dist_name == "Negative Binomial":
        guess = [12, .5, 1]
        n, p = FitNBinom(data, guess)
        return {'n': n, 'p': p}
    elif dist_name == 'Gamma':
        a, loc, scale = stats.gamma.fit(goals)
        return {'a': a, 'loc': loc, 'scale': scale}
    elif dist_name == 'Binomial':
        guess = [10, .4, 1]
        n, p = FitBinomial(data, guess)
        return {'n': int(n), 'p': p}
    
    
def add_parameter_ui(dist_name, data):
    params = dict()
    best_params = get_best_params(dist_name, data)
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
    
    # Put this into a seperate formulas.py file
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
    
    
    st.sidebar.header('Dataset Filters')
    dataset = st.sidebar.selectbox("Select Dataset", datasets)

    df = pd.read_csv(data_sources[dataset])

    seasons = st.sidebar.multiselect("Select Seasons", df.Season.unique())
    teams = st.sidebar.multiselect("Select Teams", df['Home Team'].sort_values().unique())
    venue = st.sidebar.multiselect("Select Venue", venues)
    global goals
    goals = filter_scores(df, teams, seasons, venue)['Goals']

    

    st.sidebar.header('Histogram')

    bin_size = st.sidebar.number_input('Bin Size', 1, 10, 1)
    x_max = goals.max() + 1

    if goals.size == 0:
        st.warning('No Data. Please update dataset filters.')
    else:


        global x 
        x = np.arange(0, x_max, bin_size)
        global x_arr
        x_arr = np.arange(0, x_max + bin_size, bin_size)
        global y
        y = np.histogram(goals, bins=x_arr, density=True)[0]


        st.write(f"""
        **Observations: ** {goals.count()}

        **Mean: ** {goals.mean():.2f}
        **Var: ** {goals.var():.2f}
        **Std: ** {goals.std():.2f}
        **Skew: ** {goals.skew():.2f}
        **Kurtosis: ** {goals.kurtosis():.2f}""")


        st.sidebar.header('Distribution Parameters')


        dist_name = st.sidebar.selectbox("Select Distribution", distributions)

        params = add_parameter_ui(dist_name, y)        

        st.sidebar.markdown('#### Formula')
        formula = st.sidebar.markdown(get_formula(dist_name))

        dist = get_dist_data(dist_name, params)

        # graph = plot_data(goals, dist, dist_name, params)


        graph = st.altair_chart(plot_altair(y, dist, dist_name, bin_size))

        g_data, dist_data = y, dist[:-1]


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