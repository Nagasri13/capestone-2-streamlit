import streamlit as st
import pandas as pd
import numpy as np
import io
import sklearn
from sklearn.model_selection import train_test_split
import pyod
from sklearn.preprocessing import LabelEncoder
from pyod.models.cblof import CBLOF
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.svm import OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import LocalOutlierFactor



# Libraries for missing values imputation random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer  
 # Library for missing values imputation Mice imputer
# import miceforest as mf
# Library for missing values imputation Iterative Imputer
# from fancyimpute import IterativeImputer

# # for clustering required libraries
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from kmodes.kmodes import KModes
# from kmodes.kprototypes import KPrototypes
# from sklearn.metrics import silhouette_score
# import pandas_profiling as pp
# from pandas_profiling import ProfileReport

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

# import subpackage of Matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# import 'Seaborn' 
import seaborn as sns

from scipy.stats import shapiro , pearsonr
# import statsmodels.api as sm
# from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
# from statsmodels.stats.stattools import durbin_watson
# from statsmodels.stats.api import het_breuschpagan
# from statsmodels.stats.api import het_goldfeldquandt
# from statsmodels.stats.api import jarque_bera
# from sklearn.linear_model import LinearRegression
# import statsmodels.api as sm
# from sklearn.metrics import mean_squared_error as mse
# from sklearn.model_selection import cross_val_score
# from statsmodels.stats.stattools import durbin_watson
# from sklearn.linear_model import SGDRegressor
# from scipy.stats import skew, kurtosis
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# from sklearn.svm import SVR
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# import xgboost as xgb
# import lightgbm as lgb
from scipy import stats  


st.title('NO CODE MACHINE LEARNING')

def load_data():
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write('File uploaded successfully')
        return df
    else:
        st.write('File Not uploaded')
        return None

# Sidebar
# def sidebar(df):
#     st.sidebar.title("Filter Data")
#     # Filter by columns
#     st.sidebar.subheader("Filter by Columns")
#     selected_columns = st.sidebar.multiselect("Select columns", df.columns)
#     if selected_columns:
#         df = df[selected_columns]

#     return df

# def label_encoder(df):
#     label_encoder = LabelEncoder()
#     data_encoded = df.copy()  # Create a copy of the original data to store the encoded data
#     for col in data_encoded.columns:
#         if data_encoded[col].dtype == 'object':  # Check if the column contains categorical data
#             data_encoded[col] = label_encoder.fit_transform(data_encoded[col])
#     return data_encoded

# def summarize_data(df, columns):
#     summary = pd.DataFrame(columns=['Column', 'Unique values', 'Value counts', 'Proportion of values', 'NaN count'])
#     for column in columns:
#         unique_values = df[column].astype(str).unique()
#         value_counts = df[column].value_counts()
#         proportion_of_values = df[column].value_counts(normalize=True) * 100
#         nan_count = df[column].isna().sum()
#         summary = summary.append({'Column': column,
#                                   'Unique values': unique_values,
#                                   'Value counts': value_counts,
#                                   'Proportion of values': proportion_of_values,
#                                   'NaN count': nan_count}, ignore_index=True)
#     return summary


# def groupby_aggregate(df, groupby_column, target_column):
#     agg_functions = ['mean', 'median', 'std', 'var']
#     grouped = df.groupby(groupby_column)[target_column].agg(agg_functions)
    
#     return grouped

# def normality_test(x):
#     s,p = stats.shapiro(x)
#     print(f"The statistic value is {s} with p-value : {p}")
#     if p>0.05:
#         print(f"{x.name} is normally distributed")
#     else:
#         print(f"{x.name} is not normally distributed")

# def random_forest_imputation(data_encoded):    
#     # Create an imputer
#     imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

#     # Fit the imputer to the data
#     imputer.fit(data_encoded)

#     # Transform the data
#     df_rf = imputer.transform(data_encoded)
#     df_rf = pd.DataFrame(df_rf)
#     for i in df_rf.columns: 
#         if df_rf[i].dtypes == 'float':
#             df_rf[i] = df_rf[i].astype('int')
#     df_rf.columns = data_encoded.columns
#     return df_rf

# def standard_scaler(df):
#     sc = StandardScaler()
#     scaled_df= sc.fit_transform(df)
#     return scaled_df

# def mice_imputer_imputation(data_encoded):
#     mice_imputer = data_encoded.copy()    

#     mice_imp= mf.ImputationKernel(mice_imputer,
#                              datasets = 3,
#                              save_all_iterations=True,
#                              random_state=143)
# # Run the mice algorithm for 6 iterations
#     mice_imp.mice(3)
    
#     # combine all different datasets into one final results

#     df_mice_imputer= pd.concat([mice_imp.complete_data(i) for i in range(3)]).groupby(level=0).mean()

#     return df_mice_imputer


# def iterative_imputer(data_encoded):

#     df_imputer = data_encoded.copy(deep= True)
#     iterative= IterativeImputer()
#     df_imputer.iloc[:,:] = iterative.fit_transform(df_imputer)

#     df_iterative_imputer = df_imputer.copy()

#     return df_iterative_imputer

# # Perform K-means clustering
# def perform_kmeans_clustering(df):
#     for i in df.columns: 
#         if df[i].dtypes == 'float':
#             df[i] = df[i].astype('int')
#     score = []
#     for i in range(2,10):
#         km = KMeans(n_clusters=i, random_state=100)
#         km.fit(df)
#         sil_score = silhouette_score(df, km.labels_)
#         print('silhouette score for,', i, 'clusters is ', sil_score)
#         score.append(sil_score)
#     n_clusters = score.index(max(score))+2 # Find the index of the maximum silhouette score and add 2 to get the number of clusters
#     kmeans = KMeans(n_clusters=n_clusters, random_state=100)
#     kmeans.fit(df)
#     labels = kmeans.labels_
#     df['clus_labels'] = labels
#     print('Cluster labels:', labels)
#     return df

# # Perform K-prototype clustering
# def perform_kprototype_clustering(df):
#     for i in df.columns: 
#         if df[i].dtypes == 'float':
#             df[i] = df[i].astype('int')
#     score = []
#     for i in range(2,10):
#         kp = KPrototypes(n_clusters=i, init='Cao', verbose=2)
#         clusters = kp.fit_predict(df, categorical=list(df.select_dtypes(include=['object']).columns))
#         sil_score = silhouette_score(df, clusters)
#         print('silhouette score for,', i, 'clusters is ', sil_score)
#         score.append(sil_score)
#     n_clusters = score.index(max(score)) + 2  # Find the index of the maximum silhouette score and add 2 to get the number of clusters
#     kproto = KPrototypes(n_clusters=n_clusters, init='Cao', verbose=2)
#     clusters = kproto.fit_predict(df, categorical=list(df.select_dtypes(include=['object']).columns))
#     df['clus_labels'] = clusters
#     labels = kproto.labels_
#     print('cluster labels:',labels)
#     return df

# def homoscedasticity_breuschpagan(residuals, x):
#     r, p = het_breuschpagan(residuals, x)[2:]
#     significance_level = 0.05
#     if p > significance_level:
#         st.write('Breusch-Pagan Test: We accept the null hypothesis that the errors are homoscedastic.')
#     else:
#         st.write('Breusch-Pagan Test: We accept the alternate hypothesis that the errors are heteroscedastic.')
#     return r, p


# def homoscedasticity_goldfeldquandt(residuals, x):
#     r, p, flow = het_goldfeldquandt(residuals, x)
#     significance_level = 0.05
#     if p > significance_level:
#         st.write('Goldfeld-Quandt Test: We accept the null hypothesis that the errors are homoscedastic.')
#     else:
#         st.write('Goldfeld-Quandt Test: We accept the alternate hypothesis that the errors are heteroscedastic.')
#     return r, p, flow


# def plot_jarque_bera(residuals):
#     fig, ax = plt.subplots()
#     ax.hist(residuals, bins=25)
#     ax.set_xlabel('Residuals')
#     ax.set_ylabel('Frequency')
#     st.pyplot(fig)
    

# def check_normality(x):
#     result = pd.DataFrame(columns=['Feature', 'p', 'Normality'])
#     for feature_name in x.columns:
#         r,p = shapiro(x[feature_name])
#         if p > 0.05:
#             result = result.append({'Feature': feature_name, 'p': p, 'Normality': 'yes'}, ignore_index =True)
#             f,ax1 = plt.subplots(1,1,figsize=[15,3])
#             sns.kdeplot(data=x,x=feature_name,ax=ax1)
#         else:
#             result = result.append({'Feature': feature_name, 'p': p, 'Normality': 'no'}, ignore_index =True)
#     return result

# def check_linearity(x, y):
#     if len(x) != len(y):
#         raise ValueError("x and y must have the same length")
    
#     result = pd.DataFrame(columns=['F1', 'F2', 'r', 'p', 'significant'])
#     for feature_name in x.columns:
#         if x[feature_name].dtype == "object":
#             # Convert string column to numeric
#             x[feature_name] = pd.to_numeric(x[feature_name])
#         r, p = pearsonr(x[feature_name], y)
#         if p > 0.05:
#             result = result.append({"F1": feature_name, "F2": y.name, "r": r, "p": p, "significant": "No"}, ignore_index=True)
#         else:
#             result = result.append({"F1": feature_name, "F2": y.name, "r": r, "p": p, "significant": "Yes"}, ignore_index=True)
#             fig, ax = plt.subplots()
#             ax.scatter(x[feature_name], y)
#             ax.plot(np.unique(x[feature_name]), np.poly1d(np.polyfit(x[feature_name], y, 1))(np.unique(x[feature_name])), color='red')
#             ax.set_title("Linearity plot for {} and {}".format(feature_name, y.name))
#             st.pyplot(fig)    
#     return result


# def check_multi_co_linearity(X):
#     Xc = sm.add_constant(X)
#     vif_values = [ VIF(Xc.values, i) for i in range(Xc.shape[1])]
#     DF1= pd.DataFrame(vif_values, columns = ['VIF Value'], index = Xc.columns).sort_values('VIF Value', ascending = True)
#     return DF1

# def auto_correlation(residuals):
#     a = durbin_watson(residuals)
#     return a
# def linear_regression_metrics(y_true,y_pred):
# # assume y_true and y_pred are your actual and predicted target values, respectively
# # Compute the regression metrics
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     mse = mean_squared_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     mae = mean_absolute_error(y_true, y_pred)
#     mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# # Create a dictionary to store the metrics
#     metrics = {
#     'RMSE': rmse,
#     'MSE': mse,
#     'R^2': r2,
#     'MAE': mae,
#     'MAPE': mape}
#     df_metrics = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])

# # Display the metrics table in Streamlit
#     st.write(df_metrics)

# def regression_algorithm_metrics(x_test, y_test,x_train, y_train):
#  # Define regression models
#     models = [
#     ('Linear Regression', LinearRegression()),
#     ('Ridge Regression', Ridge(alpha=0.5)),
#     ('Lasso Regression', Lasso(alpha=0.5)),
#     ('Elastic Net Regression', ElasticNet(alpha=0.5, l1_ratio=0.5)),
#     ('SVR', SVR(kernel='linear')),
#     ('Decision Tree Regression', DecisionTreeRegressor()),
#     ('Random Forest Regression', RandomForestRegressor(n_estimators=100)),
#     ('Gradient Boosting Regression', GradientBoostingRegressor()),
#     ('XGBoost Regression', xgb.XGBRegressor()),
#     ('LightGBM Regression', lgb.LGBMRegressor())
#     ]

#     # Compute metrics for each model
#     results = []
#     for name, model in models:
#         model.fit(x_train, y_train)
#         y_pred = model.predict(x_test)
#         mse = mean_squared_error(y_test, y_pred)
#         rmse = mean_squared_error(y_test, y_pred, squared=False)
#         r2 = r2_score(y_test, y_pred)
#         mae = mean_absolute_error(y_test, y_pred)
#         mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
#         results.append([name, rmse, mse, r2, mae, mape])

#     # Create dataframe of results
#     results_df = pd.DataFrame(results, columns=['Model', 'RMSE', 'MSE', 'R2', 'MAE', 'MAPE'])
#     return results_df



# def results( x_test, y_test,x_train, y_train):
#     Logit_Model = LogisticRegression()
#     Logit_Model.fit(x_train,y_train)
#     decision_tree = DecisionTreeClassifier(criterion = 'entropy')
#     decision_tree.fit(x_train,y_train)
#     random_forest = RandomForestClassifier(n_estimators=10)
#     random_forest.fit(x_train, y_train)
#     from sklearn.ensemble import AdaBoostClassifier
#     ad= AdaBoostClassifier(n_estimators = 40, random_state = 10) #creation of the object
#     ad.fit(x_train, y_train)
#     from sklearn.ensemble import GradientBoostingClassifier
#     gboost_model = GradientBoostingClassifier(n_estimators = 100, max_depth = 3, random_state = 8)
#     gboost_model.fit(x_train, y_train)
#     from xgboost import XGBClassifier
#     xgb_clf = XGBClassifier()
#     xgb_clf.fit(x_train, y_train)
#     from sklearn.naive_bayes import GaussianNB
#     gnb = GaussianNB()
#     gnb.fit(x_train, y_train)
#     models = {'logis': Logit_Model, 'decision tree': decision_tree,'random_forest': random_forest, 
#                'Adaboost': ad, 'XG Boost': xgb_clf, 'Gradient Boosting' : gboost_model, 'Naive Bayes': gnb }
#     result = pd.DataFrame(columns=['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])
#     for model_name , model in models.items():
#         y_pred = model.predict(x_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred,average='micro')
#         precision = precision_score(y_test, y_pred,average='micro')
#         recall = recall_score(y_test, y_pred,average='micro')
#         result.loc[model_name] = [type(model).__name__, accuracy, f1, precision, recall]

#     return result

 
# # Main function
# def main():
#     # Set title and subtitle
#     st.title("DATASET")

#     # Load data
#     df = load_data()

#     if df is not None:
#         # Sidebar
#         df = sidebar(df)

#         # Display data
#         st.subheader("Dataset")
#         st.write(df)
#         # Sidebar with selectbox for choosing information to display
#         option = st.sidebar.selectbox("Details of the Dataset", ('Describe', 'Shape', 'Info'))

#         # Display selected information
#         if st.button('Go and fetch the details of the dataset '):
#             if option == 'Describe':
#                 st.subheader("Dataset Description")
#                 st.write(df.describe())
#             elif option == 'Shape':
#                 st.subheader("Dataset Shape")
#                 st.write(df.shape)
#             elif option == 'Info':
#                 st.subheader("Dataset Information")
#                 with io.StringIO() as buffer:
#                     df.info(buf=buffer)
#                     info_str = buffer.getvalue()
#                     st.text(info_str)


#         # Perform data preprocessing using LabelEncoder
#         encoded_data = label_encoder(df)
        
#         # Display the encoded data
#         st.subheader("Encoded Data:")
#         if st.button('Encode the given data'):
#             st.write(encoded_data)

#         # Display data cleaning options in select box
#         data_cleaning_option = st.sidebar.selectbox("Data Cleaning", ["Missing Values", "Outliers"])

#         if data_cleaning_option == "Missing Values":
#             # Find missing values
#             st.write("### Missing Values")
#             missing_values = df.isna().sum()
#             st.write("Number of missing values in each column:")
#             st.write(missing_values)

#         elif data_cleaning_option == 'Outliers':
#             encoded_data_filled = encoded_data.fillna(0)
#             models = [OCSVM(),  KNN(), LOF(), CBLOF(), HBOS()]
#             models_name = ['OCSVM', 'KNN', 'LOF', 'CBLOF', 'IForest', 'HBOS']
#             model_dfs = {}
#             out = np.zeros(encoded_data_filled.shape[0],)
#             for i, j in zip(models, models_name):
#                 model_dfs[j] = i.fit(encoded_data_filled)
#                 out += model_dfs[j].predict(encoded_data_filled)
#             st.write('Encoded Data')
#             st.write(encoded_data)
#             st.write('outliers')
#             st.write(encoded_data_filled.iloc[np.where(out>=4)])
               
#         st.sidebar.header('Missing Value Imputation')
    
#     # Create a dropdown to select the imputation technique
#         imputation_method = st.sidebar.selectbox('Select Imputation Method', ['Random Forest', 'Iterative Imputer', 'MICE'])
#   # Perform imputation based on selected method
#         if st.button(" Impute Missing values"):
#             if imputation_method == 'Random Forest':
#                 df_imputed_ran_for = random_forest_imputation(encoded_data)
#     # Display the imputed dataframe
#                 st.write('Imputed Dataframe with Random Forest Imputer')
#                 st.write(df_imputed_ran_for)
#             elif imputation_method == 'Iterative Imputer':
#                 df_imputed_iterative_imputer = iterative_imputer(encoded_data)
#     # Display the imputed dataframe
#                 st.write('Imputed Dataframe  with iterative Imputer')
#                 st.write(df_imputed_iterative_imputer)
#             elif imputation_method == 'MICE':
#                 encoded_data = label_encoder(df)
#                 df_imputed_mice_imputer = mice_imputer_imputation(encoded_data)
#                 st.write('Imputed Dataframe  with Mice Imputer')
#                 st.write(df_imputed_mice_imputer)

#         st.sidebar.header('STANDARIZATION')
#         st.subheader('Standarized data')
#         if st.button('Dataset have Missing values click here '):
#                 df_imputed_ran_for = random_forest_imputation(encoded_data)
#                 standarized_data = standard_scaler(df_imputed_ran_for)
#                 st.write('We can use this for clustering and distance based algorithms')
#                 st.write(standarized_data)
#         elif st.button('No Missing values click here'):
#                 standarized_data = standard_scaler(encoded_data)
#                 st.write('We can use this for clustering and distance based algorithms')
#                 st.write(standarized_data)


#         st.subheader('UNSUPERVISED LEARNING - CLUSTERING')
#         st.sidebar.header('CLUSTERING')
#         st.write('No Target Variable/ dependent variable / response variable/predictor variable')
#         algorithm = st.sidebar.selectbox('Select Clustering Algorithm', ['K-means', 'K-prototype'])

#         if st.button('Missing Values imputed Dataset'):
#             if algorithm == 'K-means':
#                 encoded_data = label_encoder(df)
#                 df_imputed_ran_for = random_forest_imputation(encoded_data)
#                 k_means_cluster = perform_kmeans_clustering(df_imputed_ran_for)
#                 st.write('K_Means clustering results')
#                 st.write(k_means_cluster)
#                 st.write('Clusters count',k_means_cluster['clus_labels'].unique())

#             elif algorithm == 'K-prototype':
#                 encoded_data = label_encoder(df)
#                 df_imputed_ran_for = random_forest_imputation(encoded_data)
#                 k_proto_cluster = perform_kprototype_clustering(df_imputed_ran_for)
#                 st.write('K_Prototype clustering results')
#                 st.write(k_proto_cluster)
#                 st.write('Clusters count',k_proto_cluster['clus_labels'].unique())

#         else:
#             if st.button('No Missing Value Dataset'):
#                 if algorithm == 'K-means':
#                     encoded_data = label_encoder(df)
#                     k_means_cluster = perform_kmeans_clustering(encoded_data)
#                     st.write('K_Means clustering results')
#                     st.write(k_means_cluster)
#                     st.write('Clusters count',k_means_cluster['clus_labels'].unique())

#                 elif algorithm == 'K-prototype':
#                     encoded_data = label_encoder(df)
#                     k_proto_cluster = perform_kprototype_clustering(encoded_data)
#                     st.write('K_Prototype clustering results')
#                     st.write(k_proto_cluster)
#                     st.write('Clusters count',k_proto_cluster['clus_labels'].unique())
        
#         st.subheader('PRINCIPAL COMPONENT ANALYSIS')
#         options= st.sidebar.selectbox('Select Standarized Data For PCA',['Missing values imputed Standarized Data', 'No missing values Standarized data'] )
#         if options == 'Missing values imputed Standarized Data':
#             if st.button('Go for PCA'):
#                 df_imputed_ran_for = random_forest_imputation(encoded_data)
#                 standarized_data = standard_scaler(df_imputed_ran_for)
#                 pca = PCA(0.95)   # PCA(0.95)
#                 pc = pca.fit_transform(standarized_data)
#                 df_pc = pd.DataFrame(pc)
#                 st.write('PCA explained_variance_ratio',pca.explained_variance_ratio_ )     # variance explained by each principal component analysis
#                 st.write('Pca Dataset', df_pc)
#         elif  options == 'No missing values Standarized data':
#                  if st.button('Go For PCA'):
#                      standarized_data = standard_scaler(encoded_data)
#                      pc = pca.fit_transform(standarized_data)
#                      df_pc = pd.DataFrame(pc)
#                      st.write('Pca Dataset', df_pc)

#         if st.button('Go back to the original dataset from PCA'):
#                 df_imputed_ran_for = random_forest_imputation(encoded_data)
#                 standarized_data = standard_scaler(df_imputed_ran_for)                
#                 pca = PCA(0.95)   # PCA(0.95)
#                 pc = pca.fit_transform(standarized_data)
#                 original_dataset = pca.inverse_transform(pc)  # to go back to original dataset
#                 #st.write(original_dataset)
#                 st.write('Original dataset after PCA with missing values imputed:', original_dataset)

#         st.subheader('VISUALIZATIONS')
#         if st.button('visuals'):
#             st.subheader('Feature Analysis')
#             selected_columns = [col for col in df.columns if col != 'Unique values']
#             summary = summarize_data(df, selected_columns)
#             st.write(summary)

#         st.subheader( 'Group-by')
#         target_column = st.selectbox("Select target column", options=df.columns)
#         if st.button('Target column Mean Group_By'):
#             x = df.drop(target_column, axis=1)
#             result = df.groupby(target_column)[list(x.columns)].mean().sort_values(by=target_column, ascending=False)
#             st.write(result)


#         st.subheader('SUPERVISED LEARNING')
#         st.subheader('DATASET HAVING TARGET/DEPENDENT COLUMN GO FOR SUPERVISED LEARNING')
#         st.sidebar.header('SUPERVISED LEARNING')
#         st.subheader('Select Target Column')
#         missing = st.selectbox('Dataset Selection', ['Missing values imputed dataset','No Missing values dataset'])
#         if missing == 'Missing values imputed dataset':
#                 df_imputed_ran_for = random_forest_imputation(encoded_data)
#                 df_imputed_ran_for.columns = df.columns
#                 y = st.selectbox("Select dependent variable", df_imputed_ran_for.columns)
#                 st.write(df_imputed_ran_for[y])
#                 st.write('Your dependent Variable is Categorical So Go For Classfication Technique ',
#                       'Your dependent Variable is Continuous So Go For Regression Technique ')
#         elif missing == 'No Missing values dataset':
#                  encoded_data = label_encoder(df)
#                  y = st.selectbox("Select dependent variable", encoded_data.columns)
#                  st.write(y)
#                  st.write('Your dependent Variable is Categorical So Go For Classfication Technique ',
#                       'Your dependent Variable is Continuous So Go For Regression Technique ')
            


#         st.subheader('REGRESSION TECHNIQUES')
#         st.subheader('LINEAR REGRESSION')
#         st.sidebar.header('LINEAR REGRESSION ASSUMPTIONS BEFORE MODEL BUILDING')
#         assumptions = st.sidebar.selectbox('ASSUMPTIONS', ['Linearity', 'No Multicolinearity', 'Normality Test'])
#         if st.button('Go For Assumptions'):
#             select= st.selectbox('Select Dataset', ['Missing values imputed dataset', 'No Missing values dataset'])
#             if select == 'Missing values imputed dataset':
#                 if assumptions == 'Linearity':
#                         st.set_option('deprecation.showPyplotGlobalUse', False)
#                         df_imputed_ran_for = random_forest_imputation(encoded_data)
#                         x = df_imputed_ran_for.drop(y, axis = 1)
#                         result = check_linearity(x, df_imputed_ran_for[y])
#                         st.write("Result:")
#                         st.write(result)
#                         st.write('If there is no linearity go for other regression algorithms')
#                 elif assumptions == 'No Multicolinearity':
#                     x = df_imputed_ran_for.drop(y, axis = 1)
#                     outcome = check_multi_co_linearity(x)
#                     st.write(outcome)
#                     st.write('IF VIF Value is morethan 5 there is a strong releation between independent variables it gives no realiability so we have to reject the column',
#                         'If VIF Value is between 1 and 5 there is moderate correlation',
#                         'IF VIF value is equal to 1 there is no correlation')
#                     st.write('Solution: Go For PCA')
#                 elif assumptions == 'Normality Test':
#                     x = df_imputed_ran_for.drop(y, axis = 1)
#                     outcome = check_normality(x)
#                     st.write(outcome)
#                     st.write('If it is not normally distributed Go for TRANSFORMATIONS')
                    
#             elif select == 'No Missing values dataset':
#                 encoded_data = label_encoder(df)
#                 if assumptions == 'Linearity':
#                     x = encoded_data.drop(y, axis = 1)
#                     outcome = check_linearity(x,y)
#                     st.write(outcome)
#                     st.write('If there is no linearity go for other regression algorithms')
#                 elif assumptions == 'No Multicolinearity':
#                     x = encoded_data.drop(y, axis = 1)
#                     outcome = check_multi_co_linearity(x)
#                     st.write(outcome)
#                     st.write('IF VIF Value is morethan 5 there is a strong releation between independent variables it gives no realiability so we have to reject the column',
#                         'If VIF Value is between 1 and 5 there is moderate correlation',
#                         'IF VIF value is equal to 1 there is no correlation')
#                     st.write('Solution: Go For PCA')
#                 elif assumptions == 'Normality Test':
#                     x = encoded_data.drop(y, axis = 1)
#                     outcome = check_normality(x)
#                     st.write(outcome)
#                     st.write('If it is not normally distributed Go for TRANSFORMATIONS')
    

#         st.subheader('TRAIN TEST SPLIT')
#         st.sidebar.header('TRAIN TEST SPLIT')
#         missing_num = st.selectbox('select dataset',['Missing values imputed dataset', 'No missing values dataset'])
#         if missing_num == 'Missing values imputed dataset':
#             df_imputed_ran_for = random_forest_imputation(encoded_data)
#             X = df_imputed_ran_for.drop(y, axis = 1)
#             Y = df_imputed_ran_for[y]
#             x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3, random_state= 143)
#             splits = st.sidebar.selectbox('train test splited datas', ['x_train', 'x_test', 'y_train', 'y_test'])
#             if st.button('Show Me'):
#                 if splits == 'x_train':
#                     st.write('x_train', x_train)
#                     st.write('Shape of x_train is', x_train.shape)
#                 elif splits == 'y_train':
#                     st.write('y_train', y_train)
#                     st.write('Shape of y_train is', y_train.shape)
#                 elif splits == 'x_test':
#                     st.write('x_test', x_test)
#                     st.write('Shape of x_test is', x_test.shape)
#                 elif splits == 'y_test':
#                     st.write('y_test', y_test)
#                     st.write('Shape of y_test is', y_test.shape)  
#         elif missing_num == 'No missing values dataset':
#                 encoded_data = label_encoder(df)
#                 X = df_imputed_ran_for.drop(y, axis = 1)
#                 Y = df_imputed_ran_for[y]
#                 x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3, random_state= 143)
#                 splits = st.sidebar.selectbox('train test splited datas', ['x_train', 'x_test', 'y_train', 'y_test'])
#                 if st.button('Show Me'):
#                     if splits == 'x_train':
#                         st.write('x_train', x_train)
#                         st.write('Shape of x_train is', x_train.shape)
#                     elif splits == 'y_train':
#                         st.write('y_train', y_train)
#                         st.write('Shape of y_train is', y_train.shape)
#                     elif splits == 'x_test':
#                         st.write('x_test', x_test)
#                         st.write('Shape of x_test is', x_test.shape)
#                     elif splits == 'y_test':
#                         st.write('y_test', y_test)
#                         st.write('Shape of y_test is', y_test.shape)

#         st.subheader('LINEAR REGRESSION  MODEL BUILDING')
#         models = st.selectbox('Regression models', ['OLS', 'SGD REGRESSOR'])
#         if st.button('BUILD MODEL'):
#             selects= st.selectbox('Select Dataset', ['Missing values imputed dataset', 'No Missing values dataset'])
#             if selects == 'Missing values imputed dataset':
#                 if models == 'OLS':
#                     x = df_imputed_ran_for.drop(y, axis = 1)
#                     x = sm.add_constant(x)
#                     model=sm.OLS(df[y],x).fit()
#                     st.write('Model Summary',model.summary())
#                 elif models == 'SGD REGRESSOR':
#                     x = df_imputed_ran_for.drop(y, axis = 1)
#                     sgd = SGDRegressor(max_iter=1000, tol=1e-3, penalty='elasticnet', alpha=0.0001, random_state=42)
#                     # Fit the model to the training data
#                     sgd.fit(x_train, y_train)
#                     # Make predictions on the test data
#                     y_pred = sgd.predict(x_test)
#                     # Compute the mean squared error
#                     mse = mean_squared_error(y_test, y_pred)
#                     st.write("Mean Squared Error:", round(mse,2))

#             elif selects == 'No Missing values  dataset':
#                 if models == 'OLS':
#                     x = encoded_data.drop(y, axis = 1)
#                     x = sm.add_constant(x)
#                     model=sm.OLS(df[y],x).fit()
#                     st.write('Model Summary',model.summary())
#                 elif models == 'SGD REGRESSOR':
#                     x = encoded_data.drop(y, axis = 1)
#                     sgd = SGDRegressor(max_iter=1000, tol=1e-3, penalty='elasticnet', alpha=0.0001, random_state=42)
#                     # Fit the model to the training data
#                     sgd.fit(x_train, y_train)
#                     # Make predictions on the test data
#                     y_pred = sgd.predict(x_test)
#                     # Compute the mean squared error
#                     mse = mean_squared_error(y_test, y_pred)
#                     st.write("Mean Squared Error:", round(mse,2))

#         st.subheader('ASSUMPTIONS AFTER MODEL BUILDING')
#         if st.button('Go'):
#             st.subheader('No auto correlation')
#             selecc= st.selectbox('Select Dataset', ['Missing values imputed dataset', 'No Missing values dataset'])
#             if selecc == 'Missing values imputed dataset':
#                 x = df_imputed_ran_for.drop(y, axis = 1)
#                 model=sm.OLS(df[y],x).fit()
#                 residuals= model.resid
#                 y_pred = model.predict(x_test)
#                 fin= auto_correlation(residuals)
#                 st.write('Result:', fin)
#                 st.write('1. limits are "0 to 2" positive autocorrelation',
#                         '2. limits are equal to 2 No autocorrelation',
#                         '3. limits are equal to 2 to 4 Negative correlation')
               
#             if selecc == 'No Missing values dataset':
#                 x = encoded_data.drop(y, axis = 1)
#                 model=sm.OLS(df[y],x).fit()
#                 residuals= model.resid
#                 y_pred = model.predict(x_test)
#                 fin= auto_correlation(residuals)
#                 st.write('Result:', fin)
#                 st.write('1. limits are "0 to 2" positive autocorrelation',
#                         '2. limits are equal to 2 No autocorrelation',
#                         '3. limits are equal to 2 to 4 Negative correlation')
                
#         st.subheader("NORMALITY OF ERRORS , HOMOSCEDASTICITY TESTS, LINEAR REGRESSION MODEL METRICS")
#         if st.button('Go For'):
#             opt= st.selectbox('Select Dataset', ['Missing values imputed dataset', 'No Missing values dataset'])
#             if opt == 'Missing values imputed dataset':
#                 x = df_imputed_ran_for.drop(y, axis = 1)
#                 model=sm.OLS(df[y],x).fit()
#                 y_pred = model.predict(x_test)
#                 residuals = model.resid
#                 st.write("Skewness: ", skew(residuals))
#                 st.write("Kurtosis: ", kurtosis(residuals))
#                 plot_jarque_bera(residuals)
        
#         # Test homoscedasticity
#                 st.subheader("Homoscedasticity Tests")
#                 homoscedasticity_breuschpagan(residuals, X)
#                 homoscedasticity_goldfeldquandt(residuals, X)

                
#                 st.subheader('LINEAR REGRESSION METRICS ')

#                 model=sm.OLS(df[y],x).fit()
#                 y_pred = model.predict(x_test)
#                 linear_regression_metrics(y_test,y_pred)
#             elif opt == 'No Missing values dataset':
#                 x = encoded_data.drop(y, axis = 1)
#                 model=sm.OLS(df[y],x).fit()
#                 y_pred = model.predict(x_test)
#                 residuals = model.resid
#                 st.write("Skewness: ", skew(residuals))
#                 st.write("Kurtosis: ", kurtosis(residuals))
#                 plot_jarque_bera(residuals)
        
#                 # Test homoscedasticity
#                 st.subheader("Homoscedasticity Tests")
            
#                 homoscedasticity_breuschpagan(residuals, X)
#                 homoscedasticity_goldfeldquandt(residuals, X)

#                 st.subheader('LINEAR REGRESSION METRICS ')

#                 model=sm.OLS(df[y],x).fit()
#                 y_pred = model.predict(x_test)
#                 linear_regression_metrics(y_test,y_pred)
        
#         st.subheader('Regression Algorithms')
#         if st.button('RUN'):
#             st.write('Regression models with Metrics')
#             regression_result =  regression_algorithm_metrics(x_test, y_test,x_train, y_train)
#             st.write(regression_result)


#         st.subheader('IF THE TARGET COLUMN IS CONTINOUS GO FOR REGRESSION, IT IS CATEGORICAL GO FOR CLASSIFICATION')
#         st.subheader('CLASSIFICATION ALGORITHMS')
#         if st.button('Model Build'):
#             st.write('Classification metrics')
#             classification_result = results( x_test, y_test,x_train, y_train)
#             st.write(classification_result)

# # Run the Streamlit app
# if __name__ == '__main__':
#     main()
