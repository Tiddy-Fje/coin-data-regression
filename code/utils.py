import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from statsmodels.stats.outliers_influence import OLSInfluence
import seaborn as sb
import numpy as np


FACTOR = 1.2
SMALL_SIZE = 8 * FACTOR
MEDIUM_SIZE = 11 * FACTOR
BIGGER_SIZE = 14 * FACTOR

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#plt.rc('figure', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('figure',autolayout=True)
plt.rc('lines', linewidth=2)
plt.rc('lines', markersize=7)


def process_data( data_type = 'df-time-agg', drop_ingeborg = True, keep_coins=10, keep_persons=5, drop_lonely=False ):
    data_agg = pd.read_csv(f'../data/data-agg.csv')

    info = {}
    c_grouped_filt = data_agg.groupby('coin').filter(lambda x: x['person'].count() >= keep_coins)
    info['common_coins'] = c_grouped_filt['coin'].unique()
    p_grouped_filt = data_agg.groupby('person').filter(lambda x: x['coin'].count() >= keep_persons)
    info['common_persons'] = p_grouped_filt['person'].unique()

    data = pd.read_csv(f'../data/{data_type}.csv')
    if data_type == 'df-time-agg':
        data = data[(data['person'] != 'adamF') | (data['coin'] != '0.05EUR') | (data['agg'] != 84)]
    if drop_lonely:
        lone_coin = data_agg.groupby('coin').filter(lambda x: x['person'].count() <= 1)
        lone_person = data_agg.groupby('person').filter(lambda x: x['coin'].count() <= 1)
        data = data[~data['coin'].isin(lone_coin['coin'])]
        data = data[~data['person'].isin(lone_person['person'])]

    data['N_throws'] = data['N_start_heads_up'] + data['N_start_tails_up']
    data['tails_tails'] = data['N_start_tails_up'] - data['tails_heads']
    data['heads_tails'] = data['N_start_heads_up'] - data['heads_heads']
    data['same_side'] = data['heads_heads'] + data['tails_tails']
    data['diff_side'] = data['N_throws'] - data['same_side']

    if drop_ingeborg:
        data = data[data['person']!='ingeborgR'] # -> adds both a person and a coin (100 throws, can remove her)

    #print(len(info['common_coins']), len(info['common_persons']))
    return data, info


def deviance_analysis( data, formula, filename, summary=False, force=False ):
    def fit_model():
        model = smf.glm(formula=formula, data=data, family=sm.families.Binomial())
        results = model.fit(disp=True)
        results.save(f'../models/{filename}.pickle')
        summ = results.summary()
        with open(f'../models/{filename}.txt', 'w') as f:
            fit_history = results.fit_history
            f.write( f'Iteration:{fit_history["iteration"]}\n' ) # different use of ' and " is done
            f.write( f'Deviance trajectory:{fit_history["deviance"]}\n' ) # to avoid problems on Mac OS
            f.write(summ.as_text())
        return summ, results

    if force :
        _, results = fit_model()
    else : 
        try : 
            results = sm.load(f'../models/{filename}.pickle')
        except FileNotFoundError:
            _, results = fit_model()

    dic = {}
    dic['formula'] = formula.split('~')[1]
    dic['deviance'] = results.deviance
    dic['aic'] = results.aic
    dic['df_residual'] = results.df_resid
    dic['df_model'] = results.df_model

    if summary:
        print(results.summary())

    return dic, results


def residual_vs_covariate( results, data, ax=None, is_wls=False ):
    if ax is None : 
        fig = plt.figure(figsize=(15, 15), layout='tight')

        gs = GridSpec(11, 1, figure=fig, )
        ax1 = fig.add_subplot(gs[0:2,0])
        ax2 = fig.add_subplot(gs[3:6,0])
        ax3 = fig.add_subplot(gs[8:,0])
        ax = [ax1, ax2, ax3]

    influence_inst = OLSInfluence(results) 
    leverage = influence_inst.hat_matrix_diag
    resid_pea = results.resid_pearson / np.sqrt(1-leverage)
    if is_wls:
        davison_resid = resid_pea
    else:
        resid_dev = results.resid_deviance / np.sqrt(1-leverage)
        davison_resid = resid_dev + np.log( resid_pea / resid_dev ) / resid_dev
    

    for i, (col,lab) in enumerate( zip( ['agg', 'person', 'coin'], ['a','b','c'] ) ):
        df_resid = pd.DataFrame({'davison_resid': davison_resid, col: data[col]})
        grouped_resid = df_resid.groupby(col)
        
        if col == 'agg':
            ax[i].scatter(data[col]*100, davison_resid)
            ax[i].set_xlabel('Number of Throws')
        else:
            xticks = []
            for name, group in grouped_resid:
                sb.boxplot(x=group[col], y=group['davison_resid'], ax=ax[i])
                xticks.append(name)

            ax[i].set_xticks(xticks)
            ax[i].set_xticklabels(xticks, rotation=90)
            ax[i].set_xlabel('')
        ax[i].set_ylabel('Standardized Residuals')
        ax[i].set_title(f'$({lab})$', pad=15)

    return ax