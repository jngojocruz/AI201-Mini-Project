'''
AI 201 Mini-Project
UP Diliman

Author: Jamlech Iram N. Gojo Cruz
jngojocruz@up.edu.ph
January 2024
'''

# Python script for preparing the data
import pandas as pd             # for dataframe
import numpy as np              # for computations
from decimal import *           # for representing floating values
from matplotlib import pyplot   # for plotting

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

    age_grouping(train_data)
    feature_eng(train_data)

    #Removing columns with many NaN values
    to_remove = ['bank_interest_rate', 'mm_interest_rate', 'mfi_interest_rate', 'other_fsp_interest_rate', 'age', 'avg_shock_strength_last_year']
    train_data.drop(to_remove, inplace=True, axis=1)

    # Remove rows with NA
    train_data.dropna(inplace=True)
    #print(train_data)

    # Undersampling
    # count = train_data['classification'].value_counts()
    # np.random.seed(201)
    # drop_indices = np.random.choice(train_data[train_data['classification'] == 'poor'].index, count['poor'] - count['non-poor'], replace=False)
    # train_data.drop(drop_indices, inplace=True)
    #print(train_data['classification'].value_counts())

    # Partition to training (80%) validation (10%) and test (10%) dataset
    # random_state is used to get the same random samples on each run
    test_data = train_data.sample(frac=0.2, random_state=101)
    train_data.drop(test_data.index, inplace=True)
    validation_data = test_data.sample(frac=0.5, random_state=102)
    test_data.drop(validation_data.index, inplace=True)
    #print(train_data['classification'].value_counts())
    #print(validation_data['classification'].value_counts())
    #print(test_data['classification'].value_counts())

    return train_data, validation_data, test_data



# Grouping data by age
def age_grouping(data):
    age_condition = [
    (data['age'] < 30 ),
    (data['age'] >= 30) & (data['age'] < 45),
    (data['age'] >= 45) & (data['age'] < 60),
    (data['age'] >= 60)
    ]
    age_bins = ['< 30', '30 to 44', '45 to 60', '> 60']
    data['age_group'] = np.select(age_condition, age_bins)



# Categorical grouping
# Those categories with less representation are grouped together
# Based from: https://www.kaggle.com/code/johnnyyiu/poverty-prediction-from-visualization-to-stacking#Exploratory-data-analysis
def feature_eng(df):
    religion_categories = {'N':'N_Q', 'O':'O_P','P':'O_P', 'Q':'N_Q','X':'X'}
    df['religion'] = [religion_categories[x] for x in df['religion']]
    #print(df['religion'].value_counts())

    #num_shocks_last_year 4_5
    num_shocks_last_year_categories = {0:'0', 1:'1', 2:'2',
                        3:'3', 4:'4_5', 5:'4_5'}
    df['num_shocks_last_year'] = [num_shocks_last_year_categories[x] for x in df['num_shocks_last_year']]
    #print(df['num_shocks_last_year'].value_counts())
    
    #num_formal_institutions_last_year 3_or_over
    num_formal_institutions_last_year_categories = {0:'0', 1:'1', 2:'2',
                        3:'3_4_5_6', 4:'3_4_5_6', 5:'3_4_5_6', 6:'3_4_5_6'}
    df['num_formal_institutions_last_year'] = [num_formal_institutions_last_year_categories[x] for x in df['num_formal_institutions_last_year']]
    #print(df['num_formal_institutions_last_year'].value_counts())

    #num_informal_institutions_last_year 2_or_over
    num_informal_institutions_last_year_categories = {0:'0', 1:'1', 2:'2_3_4',
                        3:'2_3_4', 4:'2_3_4'}
    df['num_informal_institutions_last_year'] = [num_informal_institutions_last_year_categories[x] for x in df['num_informal_institutions_last_year']]
    #print(df['num_informal_institutions_last_year'].value_counts())

    relationship_to_hh_head_categories = {'Other':'Other', 'Spouse':'Spouse',
                                        'Head':'Head',
                                        'Son/Daughter':'Son/Daughter',
                                        'Sister/Brother':'Sister/Brother',
                                        'Father/Mother': 'Father/Mother',
                                        'Unknown':'Other'}
    df['relationship_to_hh_head'] = [relationship_to_hh_head_categories[x] for x in df['relationship_to_hh_head']]
    #print(df['relationship_to_hh_head'].value_counts())



# Preparing the frequency and likelihood probability tables
def get_table(train_data):
    # Group by classification (separate poor and non-poor dataset)
    grouped = train_data.groupby(train_data.classification)
    train_poor = grouped.get_group("poor")
    train_nonpoor = grouped.get_group("non-poor")
    #print(train_poor.head())
    #print(train_nonpoor.head())

    cols_insignificant = ['row_id']
    #cols_numeric = ['age','avg_shock_strength_last_year', 'num_formal_institutions_last_year', 'num_informal_institutions_last_year', 'num_financial_activities_last_year']
    # These are considered categorical: 'num_formal_institutions_last_year', 'num_informal_institutions_last_year', 'num_financial_activities_last_year' 
    #cols_numeric = ['age','avg_shock_strength_last_year']
    #cols_removed = cols_insignificant + cols_numeric + ['poverty_probability', 'classification']
    cols_removed = cols_insignificant + ['poverty_probability', 'classification']
    
    # Checker added for leave-one-out validation
    #cols_numeric = [col for col in cols_numeric if col in train_data.columns]
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
        
    # For numeric data, use normal distribution formula; get mean and stdv
    # numeric_cols_list = {}
    # for feature in train_data[cols_numeric]:
    #     classes = {}
    #     fvals_poor = train_poor[feature]
    #     fvals_nonpoor = train_nonpoor[feature]
        
    #     # Compute for class poor
    #     vals = {}
    #     vals['mean'] = np.mean(fvals_poor)
    #     vals['std'] = np.std(fvals_poor)
    #     classes['poor'] = vals

    #     # Compute for class non-poor
    #     vals = {}
    #     vals['mean'] = np.mean(fvals_nonpoor)
    #     vals['std'] = np.std(fvals_nonpoor)
    #     classes['non-poor'] = vals
        
    #     numeric_cols_list[feature] = classes

    #print(numeric_cols_list['avg_shock_strength_last_year'])
        
    return freq_table_dfs, likelihood_table_dfs #, numeric_cols_list



# Computes the normal distribution formula for numeric features
def normal_dist(x , mean , std):
    prob_density = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-((x - mean)**2)/(2 * std**2))
    return prob_density



# For evaluation of classifier
def get_measures(true_label, pred_label):
    correct_cnt = 0
    ave_sse = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in true_label.index:
        if true_label.loc[i] == pred_label.loc[i]:
            correct_cnt += 1
        else:
            ave_sse += 1
        if true_label.loc[i] == 'poor':
            if pred_label.loc[i] == 'poor':
                TP += 1
            else:
                FN += 1
        elif true_label.loc[i] == 'non-poor':
            if pred_label.loc[i] == 'non-poor':
                TN += 1
            else:
                FP += 1
    # Evaluation metrics
    accuracy = correct_cnt / len(true_label)
    ave_sse = (1/len(true_label)) * ave_sse
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f2_score = (5 * precision * recall) / (4 * precision + recall)
    specificity = TN / (TN + FP)
    balanced_acc = (recall + specificity) / 2
    
    return accuracy, ave_sse, precision, recall, f2_score, specificity, balanced_acc



# Performs the naive bayes classifier
def naive_bayes(train_data, test_data, likelihood_table_dfs, numeric_cols_list):

    # Compute P(poor) and P(non-poor)
    count = train_data['classification'].value_counts()
    P_poor = count.loc['poor'] / sum(count)
    P_nonpoor = count.loc['non-poor'] / sum(count)
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
        # for feature in numeric_cols_list:
        #     # Get the sample value as x, and compute normal dist; multiply this to previously computed likelihood prob
        #     v = test_data[feature][i]
        #     mean_poor, mean_nonpoor = numeric_cols_list[feature]['poor']['mean'], numeric_cols_list[feature]['non-poor']['mean']
        #     std_poor, std_noonpoor = numeric_cols_list[feature]['poor']['std'], numeric_cols_list[feature]['non-poor']['std']
        #     P_data_poor += Decimal(normal_dist(v, mean_poor, std_poor)).ln()
        #     P_data_nonpoor += Decimal(normal_dist(v, mean_nonpoor, std_noonpoor)).ln()
        
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
    # print(pred_labels_df)
    # print(test_data)

    # Evaluate classifier
    accuracy, ave_sse, precision, recall, f2_score, specificity, balanced_acc = get_measures(test_data['classification'], pred_labels_df['prediction'])
    
    return accuracy, ave_sse, precision, recall, f2_score, specificity, balanced_acc



# Performs k-fold validation
def cross_validation(k):
    # Get data
    train_data, validation_data, test_data = get_data()
    # Use features selected
    train_data = train_data.filter(['country','is_urban','married','education_level','literacy','employment_type_last_year','income_friends_family_last_year','income_public_sector_last_year','borrowing_recency','num_shocks_last_year','borrowed_for_emergency_last_year','can_call','can_make_transaction','phone_ownership','reg_bank_acct','active_mm_user','active_informal_nbfi_user','nonreg_active_mm_user',
                                    'row_id','poverty_probability','classification'])
    print("------------")

    # Uncomment to perform hold-out validation
    # freq_table_dfs, likelihood_table_dfs, numeric_cols_list = get_table(train_data)
    # naive_bayes(train_data, validation_data, likelihood_table_dfs, numeric_cols_list)
    
    # Number of samples per k folds
    n = len(train_data) // k
    train_data_temp = train_data
    accuracy_list = []
    f2score_list = []
    balancedacc_list = []

    for i in range(k):
        # If 2nd iteration onwards, make sure to not pick previously tested samples
        if i > 0:
            train_data_temp = train_data.drop(new_test.index)
        new_test = train_data_temp.sample(n=n)
        new_train = train_data.drop(new_test.index)
        # Get the likelihood table
        freq_table_dfs, likelihood_table_dfs = get_table(new_train)
        # Classify
        accuracy, ave_sse, precision, recall, f2_score, specificity, balanced_acc = naive_bayes(new_train, new_test, likelihood_table_dfs, None)
        #accuracy, ave_sse, precision, recall, f2_score, specificity, balanced_acc = loocv(new_train, new_test)
        print(accuracy, ave_sse, precision, recall, f2_score, specificity, balanced_acc)
        accuracy_list.append(accuracy)
        f2score_list.append(f2_score)
        balancedacc_list.append(balanced_acc)
    # Average metrics
    classifier_accuracy = sum(accuracy_list) / k
    classifier_f2score = sum(f2score_list) / k
    classifier_balancedacc = sum(balancedacc_list) / k
    
    return classifier_accuracy, classifier_f2score, classifier_balancedacc


# For feature selection, implementing leave one out method
def feature_selection():
    # Get data
    train_data, validation_data, test_data = get_data()

    # Remove insignificant features not to be tested
    #features = train_data.drop(['row_id', 'country', 'poverty_probability', 'classification'], axis='columns').columns
    #train_data.drop(['row_id','poverty_probability','classification'], axis='columns')
    #validation_data.drop(['row_id','poverty_probability','classification'], axis='columns')

    # Best features indices based on initial experiment (not included); disregard
    #best_feats = [2,0,31,16,53,30,7,35,15,6,5,23,12,11,1]
    #best_feats = [2,0,31,16,53,30,7,35,15,6]
    #best_feats = [2,0,31,16,53,30,7,35,15,6,5,23,12,11,1,27,18,17,4,3]
    #best_feats = [0,53,1]

    column_headers = list(train_data.drop(['row_id','poverty_probability','classification'], axis='columns').columns.values)
    
    #to_remove = [column_headers[i] for i in range(len(column_headers)) if i not in best_feats]
    #to_retain = [column_headers[i] for i in best_feats]
    #print(to_retain)
    #print(to_remove)
    #train_data.drop(to_remove, axis='columns', inplace=True)
    #validation_data.drop(to_remove, axis='columns', inplace=True)
    #features = to_retain
    #print(train_data.columns)

    accuracy_list = {}
    f2_list = {}

    file = open("loocv_results.txt", "a")
    file.write("Leave-One-Out\n")
    file.close()

    best_feat_acc = 0
    best_feat_f2 = 0
    best_feat_balacc = 0
    best_feat_ind = None
    len_feat = len(column_headers)

    for i in range(len(column_headers)):

        file = open("loocv_results.txt", "a")
        features = column_headers
        print(f"TOP {len(column_headers)}")
        file.write("TOP "+str(len(column_headers))+"\n")
        print(f"         {'feature':<35} {'accuracy':<22} {'ave_sse':<22} {'precision':<22} {'recall':<22} {'f2_score':<22} {'specificity':<22} {'balanced_acc'}")
        file.write('%35s %22s %22s %22s %22s %22s %22s %22s\n' % ('feature', 'accuracy', 'ave_sse', 'precision', 'recall', 'f2_score', 'specificity', 'balanced_acc'))
        file.close()

        tr_acc = 0
        tr_f2 = 0
        tr_balacc = 0
        tr = None
        cnt = 0

        for feature in features:
            # Drop the current feature
            new_train = train_data.drop(feature, axis='columns')
            new_test = validation_data.drop(feature, axis='columns')
            # Get the likelihood table
            freq_table_dfs, likelihood_table_dfs = get_table(new_train)
            # Classify
            accuracy, ave_sse, precision, recall, f2_score, specificity, balanced_acc = naive_bayes(new_train, new_test, likelihood_table_dfs, None)
            
            print(f"removed: {feature:<40} {accuracy:<22} {ave_sse:<22} {precision:<22} {recall:<22} {f2_score:<22} {specificity:<22} {balanced_acc}")
            file = open("loocv_results.txt", "a")
            file.write(f"removed: {feature:<40} {accuracy:<22} {ave_sse:<22} {precision:<22} {recall:<22} {f2_score:<22} {specificity:<22} {balanced_acc}\n")
            file.close()

            # Keep track of the metric basis
            # if accuracy > tr_acc:
            #     tr_acc = accuracy
            #     tr = cnt
            # accuracy_list[feature] = accuracy
            # cnt += 1

            if balanced_acc > tr_balacc:
                tr_f2 = f2_score
                tr_acc = accuracy
                tr_balacc = balanced_acc
                tr = cnt
            f2_list[feature] = f2_score
            cnt += 1
        
        file = open("loocv_results.txt", "a")
        file.write("\n\nREMOVED: "+str(column_headers[tr])+"\n")
        file.close()

        # Remove the feature that boosts the measure
        train_data.drop(column_headers[tr], axis='columns', inplace=True)
        validation_data.drop(column_headers[tr], axis='columns', inplace=True)
        column_headers.pop(tr)

        # Update tracker for best set of features so far
        if tr_balacc > best_feat_balacc:
            best_feat_acc = tr_acc
            best_feat_f2 = tr_f2
            best_feat_balacc = tr_balacc
            best_feat_ind = i

        file = open("loocv_results.txt", "a")
        file.write("BEST: Top "+str(len_feat - best_feat_ind)+" with balanced acc "+str(best_feat_balacc)+"\n")
        file.write("BEST: Top "+str(len_feat - best_feat_ind)+" with f2 score "+str(best_feat_f2)+"\n")
        file.write("BEST: Top "+str(len_feat - best_feat_ind)+" with accuracy "+str(best_feat_acc)+"\n")
        file.close()

    return accuracy_list, f2_list



# Performs classification using other classifiers
def loocv(train_data, test_data):
    # Train the model on training data
    X_train1 = train_data.drop(['row_id','poverty_probability', 'classification'], axis='columns')
    y_train1 = train_data['classification']
    X_test1 = test_data.drop(['row_id','poverty_probability', 'classification'], axis='columns')
    y_test1 = test_data['classification']

    # Use encoder for builtin classifiers
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    X_train = X_train1.apply(LabelEncoder().fit_transform)
    y_train = pd.DataFrame(y_train1).apply(le.fit_transform)
    # X_train = X_train.drop('classification', axis='columns')
    X_test = X_test1.apply(le.fit_transform)
    y_test = pd.DataFrame(y_test1).apply(le.fit_transform)
    # X_test = X_test.drop('classification', axis='columns')
    #print(X_train.dtypes)
    
    # Convert to numpy array
    feature_list = list(X_train.columns)
    X_train= np.array(X_train, dtype=object)
    y_train = np.array(y_train, dtype=int).ravel()
    X_test= np.array(X_test, dtype=object)
    y_test = np.array(y_test, dtype=int).ravel()
    #print(y_test, X_test)

    # Instantiate builtin classifier models
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import svm

    # Models used
    optimal_alpha = 1
    #NB_optimal = svm.SVC(kernel='linear')                                          # SVM
    #NB_optimal = DecisionTreeClassifier()                                          # Decision Tree
    #NB_optimal = LogisticRegression()                                              # Logistic Regression
    NB_optimal = KNeighborsClassifier(n_neighbors=int(np.sqrt([len(X_train)])))     # KNN
    #NB_optimal = XGBRegressor()                                                    # XGBRegressor (not included)
    
    # Fit X and y training
    NB_optimal.fit(X_train, y_train)
    
    # Predict test set
    pred = NB_optimal.predict(X_test)
    pred = pd.DataFrame(pred)
    #print(pred)

    # Inverse label encoder to 'poor' or 'non-poor'
    pred1 = le.inverse_transform(pred[0])
    pred1 = pd.DataFrame(pred1)
    y_test = pd.DataFrame(y_test)
    y_test1 = y_test1.reset_index(drop=True)
    # print(y_test1)
    # print(pred1[0])

    # Get measures
    accuracy, ave_sse, precision, recall, f2_score, specificity, balanced_acc = get_measures(y_test1, pred1[0])
    print(accuracy, ave_sse, precision, recall, f2_score, specificity, balanced_acc)

    return accuracy, ave_sse, precision, recall, f2_score, specificity, balanced_acc



# Used to predict using the test set
def predict():
    # Get train and test set
    train_data, validation_data, test_data = get_data()
    train_data = train_data.filter(['country','is_urban','married','education_level','literacy','employment_type_last_year','income_friends_family_last_year','income_public_sector_last_year','borrowing_recency','num_shocks_last_year','borrowed_for_emergency_last_year','can_call','can_make_transaction','phone_ownership','reg_bank_acct','active_mm_user','active_informal_nbfi_user','nonreg_active_mm_user',
                                    'row_id','poverty_probability','classification'])
    test_data = validation_data.filter(['country','is_urban','married','education_level','literacy','employment_type_last_year','income_friends_family_last_year','income_public_sector_last_year','borrowing_recency','num_shocks_last_year','borrowed_for_emergency_last_year','can_call','can_make_transaction','phone_ownership','reg_bank_acct','active_mm_user','active_informal_nbfi_user','nonreg_active_mm_user',
                                    'row_id','poverty_probability','classification'])
    
    # Get probability tables
    freq_table_dfs, likelihood_table_dfs = get_table(train_data)
    
    # Classify using Naive Bayes
    accuracy, ave_sse, precision, recall, f2_score, specificity, balanced_acc = naive_bayes(train_data, test_data, likelihood_table_dfs, None)

    # Classify using other classifiers
    #accuracy, ave_sse, precision, recall, f2_score, specificity, balanced_acc = loocv(train_data, test_data)

    #print(accuracy, ave_sse, precision, recall, f2_score, specificity, balanced_acc)
    return accuracy, ave_sse, precision, recall, f2_score, specificity, balanced_acc
    


# Just to remove warnings from KNN classifier
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# CROSS VALIDATION
# for k in [5,10,15,20,25,30]:
#     classifier_accuracy, classifier_f2score, classifier_balancedacc = cross_validation(k)
#     print(f"Classifier Accuracy at k={k}: {classifier_accuracy}")
#     print(f"Classifier F2 Score at k={k}: {classifier_f2score}")
#     print(f"Classifier Balanced Accuracy at k={k}: {classifier_balancedacc}")

# FEATURE SELECTION
#feature_selection()

# GETS THE AVERAGE OF THE MEASURES WITH K ITERATIONS
# k = 10
# sum_acc = 0
# sum_sse = 0
# sum_prec = 0
# sum_rec = 0
# sum_f2 = 0
# sum_spec = 0
# sum_balacc = 0
# for i in range(k):
#     accuracy, ave_sse, precision, recall, f2_score, specificity, balanced_acc = predict()
#     sum_acc+=accuracy
#     sum_sse+=ave_sse
#     sum_prec+=precision
#     sum_rec+=recall
#     sum_f2+=f2_score
#     sum_spec+=specificity
#     sum_balacc+=balanced_acc
# print("Average:")
# print(sum_acc/k, sum_sse/k, sum_prec/k, sum_rec/k, sum_f2/k, sum_spec/k, sum_balacc/k)

# PREDICT TEST SET
#predict()