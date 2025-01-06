import numpy as np
import pandas as pd

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