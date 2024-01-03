# Python script for preparing the data
import pandas as pd
import numpy as np
from decimal import *

def get_data():

    # Load dataset
    train_values = pd.read_csv('datasets/Poverty Probability Index & Economic Indicators/train_values_wJZrCmI.csv')
    train_labels = pd.read_csv('datasets/Poverty Probability Index & Economic Indicators/train_labels.csv')
    train_data = pd.concat([train_values, train_labels['poverty_probability']], axis=1)

    # Add classification based on poverty probability (y labels)
    classification = []
    for poverty_prob in train_data['poverty_probability']:
        if poverty_prob >= 0.50:
            classification.append("poor")
        else:
            classification.append("non-poor")
    train_data['classification'] = classification
    #print(train_data.head())

    #Removing columns with many NaN values
    to_remove = ['bank_interest_rate', 'mm_interest_rate', 'mfi_interest_rate', 'other_fsp_interest_rate']
    train_data.drop(to_remove, inplace=True, axis=1)

    # Remove rows with NA
    train_data.dropna(inplace=True)
    #print(train_data)

    # Partition to training (70%) validation (15%) and test (15%) dataset
    # random_state is used to get the same random samples on each run
    test_data = train_data.sample(frac=0.3, random_state=4)
    train_data.drop(test_data.index, inplace=True)
    validation_data = test_data.sample(frac=0.5, random_state=7)
    test_data.drop(validation_data.index, inplace=True)
    #print(train_data)
    #print(validation_data)
    #print(test_data)

    return train_data, validation_data, test_data

def get_table(train_data):
    # Group by classification (separate poor and non-poor dataset)
    grouped = train_data.groupby(train_data.classification)
    train_poor = grouped.get_group("poor")
    train_nonpoor = grouped.get_group("non-poor")
    #print(train_poor.head())
    #print(train_nonpoor.head())

    cols_insignificant = ['row_id', 'country']
    cols_numeric = ['age','avg_shock_strength_last_year', 'num_formal_institutions_last_year', 'num_informal_institutions_last_year', 'num_financial_activities_last_year']
    cols_removed = cols_insignificant + cols_numeric + ['poverty_probability', 'classification']
    
    # Checker added for leave-one-out validation
    cols_numeric = [col for col in cols_numeric if col in train_data.columns]
    cols_removed = [col for col in cols_removed if col in train_data.columns]

    # For categorical data
    freq_table_dfs = {}
    likelihood_table_dfs = {}
    for feature in train_data.drop(cols_removed, axis=1):
        # Frequency Table
        freq_table = pd.crosstab(train_data[feature], train_data['classification'])
        freq_table_dfs[feature] = freq_table
        #print(freq_table)

        # likelihood Probability Table
        likelihood_table = pd.DataFrame([])
        likelihood_table['non-poor'] = freq_table['non-poor'] / sum(freq_table['non-poor'])
        likelihood_table['poor'] = freq_table['poor'] / sum(freq_table['poor'])
        likelihood_table_dfs[feature] = likelihood_table
        #print(likelihood_table)

    # for p in likelihood_table_dfs:
    #     print(p)
    #     print(freq_table_dfs[p])
    #     print(likelihood_table_dfs[p])
    #     print()
        
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
        
    return freq_table_dfs, likelihood_table_dfs, numeric_cols_list


# Computes the normal distribution formula for numeric features
def normal_dist(x , mean , std):
    prob_density = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-((x - mean)**2)/(2 * std**2))
    return prob_density


# For evaluation of classifier
def get_measures(true_label, pred_label):
    correct_cnt = 0
    ave_sse = 0
    for i in true_label.index:
        if true_label.loc[i] == pred_label.loc[i]:
            correct_cnt += 1
        else:
            ave_sse += 1
    
    accuracy = correct_cnt / len(true_label)
    ave_sse = (1/len(true_label)) * ave_sse
    return accuracy, ave_sse

# Performs the naive bayes classifier
def naive_bayes(train_data, test_data, likelihood_table_dfs, numeric_cols_list):

    # Compute P(poor) and P(non-poor)
    count = train_data['classification'].value_counts()
    P_poor = count.iloc[0] / sum(count)
    P_nonpoor = count.iloc[1] / sum(count)

    #print(likelihood_table_dfs.keys())
    
    # To access: <dictionary>[<feature>][<class>][<value>]
    #print(likelihood_table_dfs['education_level']['non-poor'][1.0])
    #print(numeric_cols_list['age']['non-poor'])

    pred_labels = []
    # For each sample
    for i in test_data.index:
        P_data_poor = 0
        P_data_nonpoor = 0

        # For each feature (categorical)
        for feature in likelihood_table_dfs:
            # Get the sample value and multiply all likelihood probability with respect to that feature
            v = test_data[feature][i]
            P_data_poor += Decimal(likelihood_table_dfs[feature]['poor'][v]).ln()
            P_data_nonpoor += Decimal(likelihood_table_dfs[feature]['non-poor'][v]).ln()
        
        # For each feature (numeric)
        for feature in numeric_cols_list:
            # Get the sample value as x, and compute normal dist; multiply this to previously computed likelihood prob
            v = test_data[feature][i]
            mean_poor, mean_nonpoor = numeric_cols_list[feature]['poor']['mean'], numeric_cols_list[feature]['non-poor']['mean']
            std_poor, std_noonpoor = numeric_cols_list[feature]['poor']['std'], numeric_cols_list[feature]['non-poor']['std']
            P_data_poor += Decimal(normal_dist(v, mean_poor, std_poor)).ln()
            P_data_nonpoor += Decimal(normal_dist(v, mean_nonpoor, std_noonpoor)).ln()
        
        # Multiply likelihood prob to class prior probability
        P_poor_temp = P_data_poor + Decimal(P_poor).ln()
        P_nonpoor_temp = P_data_nonpoor + Decimal(P_nonpoor).ln()

        # Divide the resulting numerator to predictor prior probability
        P_poor_data = P_poor_temp.exp() / (P_poor_temp.exp() + P_nonpoor_temp.exp())
        P_nonpoor_data = P_nonpoor_temp.exp() / (P_poor_temp.exp() + P_nonpoor_temp.exp())

        # print(i)
        # print("P(poor|data) = ", P_poor_data)
        # print("P(nonpoor|data) = ", P_nonpoor_data)
        # print()   

        # Classify (argmax)
        label = 'poor' if P_poor_data >= P_nonpoor_data else 'non-poor'
        pred_labels.append([i, label])

    # Store predictions
    pred_labels_df = pd.DataFrame(pred_labels)
    pred_labels_df.index = pred_labels_df[0]
    pred_labels_df.columns = ['row_id', 'prediction']
    #print(pred_labels_df)
    #print(test_data)

    # Evaluate classifier
    accuracy, ave_sse = get_measures(test_data['classification'], pred_labels_df['prediction'])
    return accuracy, ave_sse


# Performs k-fold validation
def cross_validation(k):
    # Get data
    train_data, validation_data, test_data = get_data()

    # Uncomment to perform hold-out validation
    # freq_table_dfs, likelihood_table_dfs, numeric_cols_list = get_table(train_data)
    # naive_bayes(train_data, validation_data, likelihood_table_dfs, numeric_cols_list)
    
    # Number of samples per k folds
    n = len(train_data) // k
    train_data_temp = train_data
    accuracy_list = []
    for i in range(k):
        # If 2nd iteration onwards, make sure to not pick previously tested samples
        if i > 0:
            train_data_temp = train_data.drop(new_test.index)
        new_test = train_data_temp.sample(n=n, random_state=100)
        new_train = train_data.drop(new_test.index)
        # Get the likelihood table
        freq_table_dfs, likelihood_table_dfs, numeric_cols_list = get_table(new_train)
        # Classify
        accuracy, ave_sse = naive_bayes(new_train, new_test, likelihood_table_dfs, numeric_cols_list)
        print(accuracy)
        accuracy_list.append(accuracy)

    classifier_accuracy = sum(accuracy_list) / k
    return classifier_accuracy


def feature_selection():
    # Get data
    train_data, validation_data, test_data = get_data()
    # Remove insignificant features not to be tested
    features = train_data.drop(['row_id', 'country', 'poverty_probability', 'classification'], axis='columns').columns

    accuracy_list = {}
    for feature in features:
        # Drop the current feature
        new_train = train_data.drop(feature, axis='columns')
        new_test = validation_data.drop(feature, axis='columns')
        # Get the likelihood table
        freq_table_dfs, likelihood_table_dfs, numeric_cols_list = get_table(new_train)
        # Classify
        accuracy, ave_sse = naive_bayes(new_train, new_test, likelihood_table_dfs, numeric_cols_list)
        print(f"{feature}\t{accuracy}\t{ave_sse}")
        accuracy_list[feature] = accuracy

    return accuracy_list




# Test of Classifier Accuracy (Training Dataset)
# for k in [5, 10, 20]:
#     classifier_accuracy = cross_validation(k)
#     print(f"Classifier Accuracy at k={k}: {classifier_accuracy}")

#cross_validation(k=10)
#feature_selection()
