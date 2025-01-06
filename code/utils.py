import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

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
    lone_coin = data_agg.groupby('coin').filter(lambda x: x['person'].count() <= 1)
    lone_person = data_agg.groupby('person').filter(lambda x: x['coin'].count() <= 1)
    c_grouped_filt = data_agg.groupby('coin').filter(lambda x: x['person'].count() >= keep_coins)
    info['common_coins'] = c_grouped_filt['coin'].unique()
    p_grouped_filt = data_agg.groupby('person').filter(lambda x: x['coin'].count() >= keep_persons)
    info['common_persons'] = p_grouped_filt['person'].unique()

    data = pd.read_csv(f'../data/{data_type}.csv')
    if data_type == 'df-time-agg':
        data = data[(data['person'] != 'adamF') | (data['coin'] != '0.05EUR') | (data['agg'] != 84)]
    if drop_lonely:
        # remove lone coins and persons
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
            f.write( f'Iteration:{fit_history['iteration']}\n' )
            f.write( f'Deviance trajectory:{fit_history['deviance']}\n' )
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