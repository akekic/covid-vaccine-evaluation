# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: causal-covid-analysis
#     language: python
#     name: causal-covid-analysis
# ---

# # Severe Covid Cases and Vaccinations in Israel

# +
import os

import pandas as pd
import datetime

from pathlib import Path


# +
# data paths
INPUT_DIR = Path("../data/input-data")
HOSP_DATA_PATH = INPUT_DIR / 'event-among-vaccinated-120.csv'
VACCINATION_DATA_PATH = INPUT_DIR / 'vaccinated-per-day-2022-01-01.csv'
CASE_DATA_PATH = INPUT_DIR / 'cases-among-vaccinated-185.csv'
UN_POPULATION_DATA_PATH = INPUT_DIR / 'WPP2019_POP_F07_1_POPULATION_BY_AGE_BOTH_SEXES.csv'

OUTPUT_DIR = Path("../data/preprocessed-data")
OUTPUT_DIR.mkdir(exist_ok=True)
# -

# # 2 Dataset: Fine Grained Age Groups

# ## 2.1 Load Data

# ### 2.1.1 Deaths and Hospitalisations

# Remark on the vaccination status in this table: people are counted as vaccinated from the day after the dose is given. This means there is no 14 day period between dose and updated vaccination status (see event-among-vaccinated-81-readme.pdf).

df_tmp = pd.read_csv(HOSP_DATA_PATH)
df_tmp['Sunday_date'] = pd.to_datetime(df_tmp['Week'].str[:10], format='%Y-%m-%d')
df_tmp = df_tmp.drop(columns=['Week'])
# df_tmp

# + code_folding=[0, 6, 18]
col_map_hosp = {
    'event_after_1st_dose': 'hosp_after_1st_dose',
    'event_after_2nd_dose': 'hosp_after_2nd_dose',
    'event_after_3rd_dose': 'hosp_after_3rd_dose',
    'event_for_not_vaccinated': 'hosp_unvaccinated',
}
col_map_death = {
    'event_after_1st_dose': 'death_after_1st_dose',
    'event_after_2nd_dose': 'death_after_2nd_dose',
    'event_after_3rd_dose': 'death_after_3rd_dose',
    'event_for_not_vaccinated': 'death_unvaccinated',
}
df = df_tmp[df_tmp['Type_of_event'] == 'Hospitalization'].rename(columns=col_map_hosp).drop(columns='Type_of_event')

df = df.merge(
    df_tmp[df_tmp['Type_of_event'] == 'Death'].rename(columns=col_map_death).drop(columns='Type_of_event'),
    on=['Age_group', 'Sunday_date']
)
cols = [
    'Sunday_date',
    'Age_group',
    'hosp_after_1st_dose',
    'hosp_after_2nd_dose',
    'hosp_after_3rd_dose',
    'hosp_unvaccinated',
    'death_after_1st_dose',
    'death_after_2nd_dose',
    'death_after_3rd_dose',
    'death_unvaccinated',
]
df = df[cols]
# df

# + code_folding=[0]
clean_cols =[
    'hosp_after_1st_dose',
    'hosp_after_2nd_dose',
    'hosp_after_3rd_dose',
    'hosp_unvaccinated',
    'death_after_1st_dose',
    'death_after_2nd_dose',
    'death_after_3rd_dose',
    'death_unvaccinated',
]
data_cleaning_map = {pd.NA: '0', '<5': '1'}  # TODO: check if <5 has to be mapped to sth else
df[clean_cols] = df[clean_cols].replace(data_cleaning_map).astype(float).astype(int)
df['hosp_total'] = (
    df['hosp_unvaccinated'] + df['hosp_after_1st_dose'] + df['hosp_after_2nd_dose'] + df['hosp_after_3rd_dose']
)
df['death_total'] = (
    df['death_unvaccinated'] + df['death_after_1st_dose'] + df['death_after_2nd_dose'] + df['death_after_3rd_dose']
)
df.head()
# -

# ### 2.1.2 Vaccinations

# + code_folding=[2]
df_vac = pd.read_csv(VACCINATION_DATA_PATH)

clean_cols = [
    'first_dose',
    'second_dose',
    'third_dose',
]
data_cleaning_map = {'<15': '10'}  # TODO: check if <15 has to be mapped to sth. else
df_vac[clean_cols] = df_vac[clean_cols].replace(data_cleaning_map).astype(int)

df_vac['Sunday_date'] = pd.to_datetime(
    df_vac['VaccinationDate'], format='%Y-%m-%d'
).dt.to_period('W-SAT').apply(lambda r: r.start_time) # Sunday-Monday weeks

df_vac = df_vac.rename(columns={'first_dose': '1st_dose', 'second_dose': '2nd_dose', 'third_dose': '3rd_dose'})
# df_vac
# -

# merge vaccination data
df_vac_grouped = df_vac.groupby(['Sunday_date', 'age_group']).sum().reset_index()
df = df.merge(df_vac_grouped.rename(columns={'age_group': 'Age_group'}), on=['Sunday_date', 'Age_group'])
df.head()

# ### 2.1.3 Demographic Data

# +
# load UN population data

# convert to csv, uncomment to redo
# df_pop = pd.read_excel(
#     'data/israel/WPP2019_POP_F07_1_POPULATION_BY_AGE_BOTH_SEXES.xlsx',
#     skiprows=15,
#     header=1,
#     index_col=0
# )
# df_pop = df_pop[
#     (df_pop['Region, subregion, country or area *'] == 'Israel')
#     & (df_pop['Reference date (as of 1 July)'] == 2020)
# ]

# df_pop.to_csv('data/israel/WPP2019_POP_F07_1_POPULATION_BY_AGE_BOTH_SEXES.csv')

df_pop = pd.read_csv(UN_POPULATION_DATA_PATH, index_col=0)

df_pop

# + code_folding=[10]
df_pop['0-19'] = 1000 * (df_pop['0-4'] + df_pop['5-9'] + df_pop['10-14'] + df_pop['15-19']).astype(int)
df_pop['20-29'] = 1000 * (df_pop['20-24'] + df_pop['25-29']).astype(int)
df_pop['30-39'] = 1000 * (df_pop['30-34'] + df_pop['35-39']).astype(int)
df_pop['40-49'] = 1000 * (df_pop['40-44'] + df_pop['45-49']).astype(int)
df_pop['50-59'] = 1000 * (df_pop['50-54'] + df_pop['55-59']).astype(int)
df_pop['60-69'] = 1000 * (df_pop['60-64'] + df_pop['65-69']).astype(int)
df_pop['70-79'] = 1000 * (df_pop['70-74'] + df_pop['75-79']).astype(int)
df_pop['80-89'] = 1000 * (df_pop['80-84'] + df_pop['85-89']).astype(int)
df_pop['90+'] = 1000 * (df_pop['90-94'] + df_pop['95-99'] + df_pop['100+']).astype(int)

drop_cols = [
    'Variant',
    'Region, subregion, country or area *',
    'Notes',
    'Country code',
    'Type',
    'Parent code',
    'Reference date (as of 1 July)',
    '0-4',
    '5-9',
    '10-14',
    '15-19',
    '20-24',
    '25-29',
    '30-34',
    '35-39', 
    '40-44',
    '45-49',
    '50-54',
    '55-59',
    '60-64',
    '65-69',
    '70-74',
    '75-79',
    '80-84',
    '85-89',
    '90-94',
    '95-99',
    '100+',
]
df_pop = df_pop.drop(columns=drop_cols).rename(
    index={1440: 'Population_size'}).T.reset_index().rename(columns={'index': 'Age_group'})
# df_pop

# +
# rescale by current population
# source: https://www.cbs.gov.il/he/mediarelease/DocLib/2020/438/11_20_438e.pdf
current_population = 9291000
df_pop['Population_size'] = df_pop['Population_size'] * (current_population / df_pop['Population_size'].sum())
df_pop['Population_share'] = df_pop['Population_size']/current_population

df_constants = pd.DataFrame({'current_population': [current_population]})
# df_pop
# -

# merge population data into main datafram
df = df.merge(df_pop, on=['Age_group'])
df.head()

# ### 2.1.4 Infection Data

# + code_folding=[]
df_inf = pd.read_csv(CASE_DATA_PATH)
df_inf['Sunday_date'] = pd.to_datetime(df_inf['Week'].str[:10], format='%Y-%m-%d')
df_inf = df_inf.drop(columns=['Week'])

clean_cols = [
    'positive_1_6_days_after_1st_dose',
    'positive_7_13_days_after_1st_dose',
    'positive_14_20_days_after_1st_dose',
    'positive_above_20_days_after_1st_dose',
    'positive_1_6_days_after_2nd_dose',
    'positive_7_13_days_after_2nd_dose',
    'positive_14_30_days_after_2nd_dose',
    'positive_31_90_days_after_2nd_dose',
    'positive_above_3_month_after_2nd_before_3rd_dose',
    'positive_1_6_days_after_3rd_dose',
    'positive_7_13_days_after_3rd_dose',
    'positive_14_30_days_after_3rd_dose',
    'positive_31_90_days_after_3rd_dose',
    'positive_above_90_days_after_3rd_dose',
    'Sum_positive_without_vaccination',
]

data_cleaning_map = {pd.NA: '0', '<5': '4'}  # TODO: check if <5 has to be mapped to sth else
df_inf[clean_cols] = df_inf[clean_cols].replace(data_cleaning_map).astype(float).astype(int)

sum_total_cols = [
    'positive_1_6_days_after_1st_dose',
    'positive_7_13_days_after_1st_dose',
    'positive_14_20_days_after_1st_dose',
    'positive_above_20_days_after_1st_dose',
    'positive_1_6_days_after_2nd_dose',
    'positive_7_13_days_after_2nd_dose',
    'positive_14_30_days_after_2nd_dose',
    'positive_31_90_days_after_2nd_dose',
    'positive_above_3_month_after_2nd_before_3rd_dose',
    'positive_1_6_days_after_3rd_dose',
    'positive_7_13_days_after_3rd_dose',
    'positive_14_30_days_after_3rd_dose',
    'positive_31_90_days_after_3rd_dose',
    'positive_above_90_days_after_3rd_dose',
    'Sum_positive_without_vaccination',
]
df_inf['positive_total'] = df_inf[sum_total_cols].sum(axis=1)

sum_unvaccinated_cols = [
    'Sum_positive_without_vaccination',
]
df_inf['positive_unvaccinated'] = df_inf[sum_unvaccinated_cols].sum(axis=1)

sum_1st_cols = [
    'positive_1_6_days_after_1st_dose',
    'positive_7_13_days_after_1st_dose',
    'positive_14_20_days_after_1st_dose',
    'positive_above_20_days_after_1st_dose',
]
df_inf['positive_after_1st_dose'] = df_inf[sum_1st_cols].sum(axis=1)

sum_2nd_cols = [
    'positive_1_6_days_after_2nd_dose',
    'positive_7_13_days_after_2nd_dose',
    'positive_14_30_days_after_2nd_dose',
    'positive_31_90_days_after_2nd_dose',
    'positive_above_3_month_after_2nd_before_3rd_dose',
]
df_inf['positive_after_2nd_dose'] = df_inf[sum_2nd_cols].sum(axis=1)

sum_3rd_cols = [
    'positive_1_6_days_after_3rd_dose',
    'positive_7_13_days_after_3rd_dose',
    'positive_14_30_days_after_3rd_dose',
    'positive_31_90_days_after_3rd_dose',
    'positive_above_90_days_after_3rd_dose',
]
df_inf['positive_after_3rd_dose'] = df_inf[sum_3rd_cols].sum(axis=1)

df_inf['Population_share_pos'] = df_inf[
    ['Sunday_date', 'Age_group', 'positive_total']
].groupby(['Sunday_date']).apply(lambda x: x['positive_total']/x['positive_total'].sum()).values

df_inf.head()

# + code_folding=[0]
merge_cols = [
    'Sunday_date', 
    'Age_group', 
    'positive_total',
    'positive_unvaccinated', 
    'positive_after_1st_dose',
    'positive_after_2nd_dose', 
    'positive_after_3rd_dose',
    'Population_share_pos',
]
df = df.merge(df_inf[merge_cols], on=['Sunday_date', 'Age_group'], how='left')
# df
# -

# ### 2.1.5 Data Preparation

# + code_folding=[2]
# Aggregated numbers over all age groups
df_tmp = df.groupby('Sunday_date').agg(
    {
        'hosp_unvaccinated': 'sum',
        'hosp_after_1st_dose': 'sum',
        'hosp_after_2nd_dose': 'sum',
        'hosp_after_3rd_dose': 'sum',
        'hosp_total': 'sum',
        'death_unvaccinated': 'sum',
        'death_after_1st_dose': 'sum',
        'death_after_2nd_dose': 'sum',
        'death_after_3rd_dose': 'sum',
        'death_total': 'sum',
        '1st_dose': 'sum',
        '2nd_dose': 'sum',
        '3rd_dose': 'sum',
        'Population_size': 'sum',
        'Population_share': 'sum',
        'Population_share_pos': 'sum',
        'positive_total': 'sum',
        'positive_unvaccinated': 'sum', 
        'positive_after_1st_dose': 'sum',
        'positive_after_2nd_dose': 'sum', 
        'positive_after_3rd_dose': 'sum',
    }
).reset_index()
df_tmp['Age_group'] = 'total'
df = pd.concat([df, df_tmp]).sort_values(['Sunday_date', 'Age_group']).reset_index()
# df.head()

# +
# Number of vaccine doses relative to population size
df['1st_dose_rel'] = df['1st_dose']/df['Population_size']
df['2nd_dose_rel'] = df['2nd_dose']/df['Population_size']
df['3rd_dose_rel'] = df['3rd_dose']/df['Population_size']

# df.head()

# +
# Cumulative number of vaccine doses for each week
cols = ['Sunday_date', 'Age_group', '1st_dose', '2nd_dose', '3rd_dose']
col_map = {
    '1st_dose': '1st_dose_cum',
    '2nd_dose': '2nd_dose_cum',
    '3rd_dose': '3rd_dose_cum',
}

df = df.join(df[cols].groupby(['Age_group']).transform('cumsum').rename(columns=col_map))
df['unvaccinated_cum'] = df['Population_size'] - df['1st_dose_cum']

for col in ['1st_dose', '2nd_dose', '3rd_dose', 'unvaccinated']:
    df[col + '_cum_rel'] = df[col + '_cum'] / df['Population_size']

df.head()

# +
# Share of vaccinated under infected population
df['unvaccinated_rel_pos'] = df['positive_unvaccinated'] / df['positive_total']
df['1st_dose_rel_pos'] = df['positive_after_1st_dose'] / df['positive_total']
df['2nd_dose_rel_pos'] = df['positive_after_2nd_dose'] / df['positive_total']
df['3rd_dose_rel_pos'] = df['positive_after_3rd_dose'] / df['positive_total']

# df.head()

# + code_folding=[]
# Hospitalisation and death rates
df['hosp_unvaccinated_rel'] = df['hosp_unvaccinated']/df['unvaccinated_cum']
df['hosp_after_1st_dose_rel'] = df['hosp_after_1st_dose']/(df['1st_dose_cum'] - df['2nd_dose_cum'])
df['hosp_after_2nd_dose_rel'] = df['hosp_after_2nd_dose']/(df['2nd_dose_cum'] - df['3rd_dose_cum'])
df['hosp_after_3rd_dose_rel'] = df['hosp_after_3rd_dose']/df['3rd_dose_cum']
df['hosp_total_rel'] = df['hosp_total']/df['Population_size']

df['death_unvaccinated_rel'] = df['death_unvaccinated']/df['unvaccinated_cum']
df['death_after_1st_dose_rel'] = df['death_after_1st_dose']/(df['1st_dose_cum'] - df['2nd_dose_cum'])
df['death_after_2nd_dose_rel'] = df['death_after_2nd_dose']/(df['2nd_dose_cum'] - df['3rd_dose_cum'])
df['death_after_3rd_dose_rel'] = df['death_after_3rd_dose']/df['3rd_dose_cum']
df['death_total_rel'] = df['death_total']/df['Population_size']

# detected infection rate
df['positive_unvaccinated_rel'] = df['positive_unvaccinated']/df['unvaccinated_cum']
df['positive_after_1st_dose_rel'] = df['positive_after_1st_dose']/(df['1st_dose_cum'] - df['2nd_dose_cum'])
df['positive_after_2nd_dose_rel'] = df['positive_after_2nd_dose']/(df['2nd_dose_cum'] - df['3rd_dose_cum'])
df['positive_after_3rd_dose_rel'] = df['positive_after_3rd_dose']/df['3rd_dose_cum']
df['positive_total_rel'] = df['positive_total']/df['Population_size']

# TODO: check why there are rates > 1 (maybe related to <5 mapping)
# df['hosp_unvaccinated_rel_pos'] = (df['hosp_unvaccinated']/df['positive_unvaccinated'].shift(0)).clip(upper=1)
# df['hosp_after_1st_dose_rel_pos'] = (df['hosp_after_1st_dose']/df['positive_after_1st_dose'].shift(0)).clip(upper=1)
# df['hosp_after_2nd_dose_rel_pos'] = (df['hosp_after_2nd_dose']/df['positive_after_2nd_dose'].shift(0)).clip(upper=1)
# df['hosp_after_3rd_dose_rel_pos'] = (df['hosp_after_3rd_dose']/df['positive_after_3rd_dose'].shift(0)).clip(upper=1)
# df['hosp_total_rel_pos'] = (df['hosp_total']/df['positive_total'].shift(0)).clip(upper=1)

df['hosp_unvaccinated_rel_pos'] = (df['hosp_unvaccinated']/df['positive_unvaccinated'].shift(0))
df['hosp_after_1st_dose_rel_pos'] = (df['hosp_after_1st_dose']/df['positive_after_1st_dose'].shift(0))
df['hosp_after_2nd_dose_rel_pos'] = (df['hosp_after_2nd_dose']/df['positive_after_2nd_dose'].shift(0))
df['hosp_after_3rd_dose_rel_pos'] = (df['hosp_after_3rd_dose']/df['positive_after_3rd_dose'].shift(0))
df['hosp_total_rel_pos'] = (df['hosp_total']/df['positive_total'].shift(0))

# df['death_unvaccinated_rel_pos'] = (df['death_unvaccinated']/df['positive_unvaccinated']).clip(upper=1)
# df['death_after_1st_dose_rel_pos'] = (df['death_after_1st_dose']/df['positive_after_1st_dose']).clip(upper=1)
# df['death_after_2nd_dose_rel_pos'] = (df['death_after_2nd_dose']/df['positive_after_2nd_dose']).clip(upper=1)
# df['death_after_3rd_dose_rel_pos'] = (df['death_after_3rd_dose']/df['positive_after_3rd_dose']).clip(upper=1)
# df['death_total_rel_pos'] = (df['death_total']/df['positive_total']).clip(upper=1)

df['death_unvaccinated_rel_pos'] = (df['death_unvaccinated']/df['positive_unvaccinated'])
df['death_after_1st_dose_rel_pos'] = (df['death_after_1st_dose']/df['positive_after_1st_dose'])
df['death_after_2nd_dose_rel_pos'] = (df['death_after_2nd_dose']/df['positive_after_2nd_dose'])
df['death_after_3rd_dose_rel_pos'] = (df['death_after_3rd_dose']/df['positive_after_3rd_dose'])
df['death_total_rel_pos'] = (df['death_total']/df['positive_total'])

df.head()
# -

# ### Delta and Alpha Variant Periods

# +
df_constants['delta_0_date_from'] = datetime.datetime(year=2021, month=1, day=17)
df_constants['delta_0_date_to'] = datetime.datetime(year=2021, month=3, day=14)

df_constants['delta_1_date_from'] = datetime.datetime(year=2021, month=8, day=1)
df_constants['delta_1_date_to'] = datetime.datetime(year=2021, month=9, day=26)

df_constants
# -

# ### Save to Disc

# save to disc
df.to_pickle(OUTPUT_DIR / 'israel_df.pkl')
df_constants.to_pickle(OUTPUT_DIR / 'israel_constants.pkl')
