# Python script for preparing the data
import pandas as pd
import numpy as np

# Load dataset
train_values = pd.read_csv('datasets/Poverty Probability Index & Economic Indicators/train_values_wJZrCmI.csv')
train_labels = pd.read_csv('datasets/Poverty Probability Index & Economic Indicators/train_labels.csv')
train_data = pd.concat([train_values, train_labels], axis=1)

# Add classification based on poverty probability (y labels)
classification = []
for poverty_prob in train_data['poverty_probability']:
    if poverty_prob >= 0.50:
        classification.append("poor")
    else:
        classification.append("non-poor")
train_data['classification'] = classification
#print(train_data.head())

# Group by classification (separate poor and non-poor dataset)
grouped = train_data.groupby(train_data.classification)
train_poor = grouped.get_group("poor")
train_nonpoor = grouped.get_group("non-poor")
#print(train_poor.head())
#print(train_nonpoor.head())

cols_insignificant = ['row_id', 'country']
cols_numeric = ['bank_interest_rate', 'mm_interest_rate', 'mfi_interest_rate', 'other_fsp_interest_rate', 'avg_shock_strength_last_year', 'num_formal_institutions_last_year', 'num_informal_institutions_last_year', 'num_financial_activities_last_year']
cols_removed = cols_insignificant + cols_numeric

# For categorical data
freq_table_dfs = {}
posterior_table_dfs = {}
for feature in train_data.drop(cols_removed, axis=1):
    # Frequency Table
    freq_table = pd.crosstab(train_data[feature], train_data['classification'])
    freq_table_dfs[feature] = freq_table
    #print(freq_table)

    # Posterior Probability Table
    posterior_table = pd.DataFrame([])
    posterior_table['non-poor'] = freq_table['non-poor'] / sum(freq_table['non-poor'])
    posterior_table['poor'] = freq_table['poor'] / sum(freq_table['poor'])
    posterior_table_dfs[feature] = posterior_table
    #print(posterior_table)

#print(posterior_table_dfs['education_level'])
    
# For numeric data
numeric_cols_list = {}
for feature in train_data[cols_numeric]:
    classes = {}
    fvals_poor = train_poor[feature]
    fvals_nonpoor = train_nonpoor[feature]
    
    # Compute for class poor
    vals = {}
    vals['mean'] = np.mean(fvals_poor)
    vals['std'] = np.std(fvals_poor)
    classes['poor'] = vals

    # Compute for class non-poor
    vals = {}
    vals['mean'] = np.mean(fvals_nonpoor)
    vals['std'] = np.std(fvals_nonpoor)
    classes['non-poor'] = vals
    
    numeric_cols_list[feature] = classes

#print(numeric_cols_list['avg_shock_strength_last_year'])