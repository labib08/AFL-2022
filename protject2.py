import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import probplot


DATA_FILENAME = 'AFL-2022-totals.csv'

def project2(filename):
    BROWN_VOTES = 'BR'
    SPLIT = 0.2
    THRESHOLD = 0.3
    SIG_FIG = 4
   

    # Reads the csv file.
    df = pd.read_csv(filename)

    # Fills up the missing values in csv dataset with 0.
    df = df.fillna(0)

    # Remove "Player" and "Team" features
    df = df.iloc[:, 2:]

    # Gets the data features as a list
    data_features = list(df.columns)

    
    # Get independent features (features exluding brownlow votes)
    X = df.loc[:, df.columns != BROWN_VOTES]
    # Get dependent feature (brownlow votes)
    y = df[BROWN_VOTES]

    # Perform train-test split
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=SPLIT, random_state=42)
    X_train_features = list(X_train.columns)


    # Produce Table of Mutual Information and Pearson Correlation for each feature &
    # Graphs for each feature against Brownlow Votes

    correlation_dict = {}
    correlation_dict["Correlation"] = ["Pearson", "Mutual Information"]

    for data in X_train_features:

        # correlations
        correlation_list = []
        pearson_coefficient = round(pearson_r(X_train[data], y_train), SIG_FIG)
        mutual_info = round(normalized_mutual_info_score(X_train[data], y_train, average_method='min'), SIG_FIG)
        correlation_list.append(pearson_coefficient)
        correlation_list.append(mutual_info)
        correlation_dict[data] = correlation_list

        # graph features against brownlow votes
        plt.scatter(X_train[data], y_train)
        decode = {
        'GM': 'Games played','KI': 'Kicks','MK': 'Marks','HB': 'Handballs',
        'DI': 'Disposals','GL': 'Goals','BH': 'Behinds','HO': 'Hit outs',
        'TK': 'Tackles','RB': 'Rebound SOs','IF': 'Inside SOs','CL': 'Clearances',
        'CG': 'Clangers','FF': 'Free kicks for','FA': 'Free kicks against',
        'BR': 'Brownlow votes','CP': 'Contested possessions','UP': 'Uncontested possessions',
        'CM': 'Contested marks','MI': 'Marks inside SO','1%': 'One percenters',
        'BO': 'Bounces','GA': 'Goal assist'
        }
        plt.title(f"{decode[data]} vs Brownlow Votes")
        if data == "1%":
            data = "1 percent"
        plt.xlabel(f"{data}")
        plt.ylabel("BR")
        plt.savefig(f"{data} vs BR.png")
        plt.close()

    output_correlation_df = pd.DataFrame(correlation_dict)
    print("\n\nPearson Correlation and Mutual Information for Each Feature, With Respect to Brownlow Votes\n")
    print(output_correlation_df.to_string(index=False))

    # Drop features with MI below Threshold
    drop_least_corr(X_train, X_test, THRESHOLD, y_train)

    # Generate table for collinearity of remaining features
    print("\n\n\nTable to Assess Collinearity of Each Feature:\n")
    print(X_train.corr())

    # SUPERVISED LEARNING

    # Scale data for Linear Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Perform Linear Regression
    reg = LinearRegression().fit(X_train_scaled,y_train)
    reg.fit(X_train_scaled, y_train)

    # Test Linear Regression
    r2_train_reg = reg.score(X_train_scaled, y_train)
    r2_test_reg = reg.score(X_test_scaled, y_test)
    y_pred_reg = reg.predict(X_test_scaled)
    mse_reg = MSE(y_test, y_pred_reg)
    
    # Test using 10-Fold validtaion
    nf_CV = KFold(n_splits=10, shuffle = True, random_state=42)
    r2_cross_val_reg = cross_val_score(reg, X, y, cv=nf_CV, scoring = 'r2')
    mse_cross_val_reg =  -cross_val_score(reg, X, y, cv=nf_CV, scoring='neg_mean_squared_error').mean()

    # Generate Q-Q plot to test normality of residuals
    probplot(y_test - y_pred_reg, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Residuals")
    plt.savefig("Q-Q Plot")


    # Now do Random Forest
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    
    # Test Random Forest
    r2_train_rf = rf.score(X_train, y_train)
    r2_test_rf = rf.score(X_test, y_test)
    y_pred_rf = rf.predict(X_test)
    mse_rf = MSE(y_test, y_pred_rf)

    # Perform 10-Fold cross validation with Random Forest
    r2_cross_val_rf = cross_val_score(rf, X, y, cv=nf_CV, scoring='r2')
    mse_cross_val_rf =  -cross_val_score(rf, X, y, cv=nf_CV, scoring='neg_mean_squared_error').mean()


    # Output Results
    output_model = {
        "Model": ["Linear Regression", "Regression Forrest"],
        "Training R^2": [r2_train_reg, r2_train_rf],
        "Test R^2": [r2_test_reg, r2_test_rf],
        "Test MSE": [mse_reg, mse_rf],
        "10-fold-CV R^2": [r2_cross_val_reg.mean(), r2_cross_val_rf.mean()],
        "10-fold-CV MSE": [mse_cross_val_reg, mse_cross_val_rf]
        
    }
    output_model_df = pd.DataFrame(output_model)
    print("\n\n\n Results of Supervised Learning Models:\n")
    print(output_model_df.to_string(index=False))




def drop_least_corr(X_train, X_test, cutoff, y_train):
    '''
    Remove the columns with the least correlations (mutual information) with Brownlow Votes 
    '''
    mi_scores = X_train.apply(lambda x: normalized_mutual_info_score(x, y_train, average_method='min'))
    dropped_stats = X_train.columns[mi_scores < cutoff].tolist()
    X_train.drop(columns=dropped_stats, inplace=True)
    X_test.drop(columns=dropped_stats, inplace=True)



def pearson_r(feature_a, feature_b):
    """
    A function which computes the Pearson Correlation between two features. This function is implemented from
    workshop Ed Lessons from the topic "Correlation".
    """
    # compute the mean
    mean_a = feature_a.mean()
    mean_b = feature_b.mean()
    
    # compute the numerator of pearson r
    numerator = sum((feature_a - mean_a) * (feature_b - mean_b))
    
    # compute the denominator of pearson r
    denominator = np.sqrt(sum((feature_a - mean_a) ** 2) * sum((feature_b - mean_b) ** 2))
    
    return numerator/denominator

project2(DATA_FILENAME)
