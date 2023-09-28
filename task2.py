import os
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import logit
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.tools import add_constant
from itertools import product


SHOW_PLOTS = True
INTERACTIVE = True

# Create output2 directory
os.makedirs('./output2', exist_ok=True)

# Load data
medical_clean = pd.read_csv(r"C:\Users\user\OneDrive\Documents\Education\Western Govenors University"
                            r"\MS - Data Analytics\D208 - Predictive Modeling\medical_clean.csv")

# Select dependent variable (continuous)
# and all potential independent variables (at least one categorical)
# Dependent = ReAdmis
# Independent: Everything that could be relevant to start
columns = ['ReAdmis', 'Lat', 'Lng', 'Children', 'Age', 'Income', 'VitD_levels', 'Doc_visits', 'Full_meals_eaten',
           'vitD_supp', 'Initial_days', 'TotalCharge', 'Additional_charges', 'HighBlood', 'Stroke', 'Overweight',
           'Arthritis', 'Diabetes', 'Hyperlipidemia', 'BackPain', 'Anxiety', 'Allergic_rhinitis', 'Reflux_esophagitis',
           'Asthma', 'Area', 'Marital', 'Gender', 'Initial_admin', 'Complication_risk', 'Services']
df = medical_clean[columns]

print('\n#---------------#\n'
      '| DATA CLEANING |\n'
      '#---------------#\n')
# Duplicate detection
print(f'Duplication check:\n\tNumber of duplicated rows: {df.duplicated().sum()}')
input('Enter to continue...\n') if INTERACTIVE else print()

# Null Detection/Handling
print(f'Null detection:\n{df.isna().sum()}')
input('Enter to continue...\n') if INTERACTIVE else print()

# Outlier Detection/Handling
num_cols = [column for column in columns if df[column].dtype in ['float64', 'int64']]
df_num = df[num_cols]
df_zscores = df_num.apply(stats.zscore)
df_outliers = df_zscores.apply(lambda x: (x > 3) | (x < -3))
print(f'Outlier counts:\n{df_outliers.sum().to_string()}')
input('Enter to continue...\n') if INTERACTIVE else print()

# Investigate Location outliers
location_outliers = medical_clean.loc[df_outliers.Lat | df_outliers.Lng, ['State', 'Lat', 'Lng']]
location_outliers.columns = ['State', 'Lat_mean', 'Lng_mean']
location_outlier_groups = location_outliers.groupby("State", as_index=False).mean()
print(f'Location Outliers:\n{location_outlier_groups.to_string()}')
input('Enter to continue...\n') if INTERACTIVE else print()

# Investigate Children outliers
children_outliers = medical_clean.loc[df_outliers.Children, ['Children', 'Customer_id']]
children_outliers.columns = ['Children', 'count']
children_outliers = children_outliers.groupby("Children", as_index=False).count()
print(f'Children Outliers:\n{children_outliers.to_string()}')
input('Enter to continue...\n') if INTERACTIVE else print()

# Investigate Income outliers
income_outliers = medical_clean.loc[df_outliers.Income, ['Job', 'Income']]
income_outliers.columns = ['Job', 'Income_mean']
income_outliers = income_outliers.groupby("Job", as_index=False).mean(). \
    sort_values("Income_mean", ascending=False)
print(f'Income Outliers:\n{income_outliers.iloc[:10].to_string()}')
input('Enter to continue...\n') if INTERACTIVE else print()

# Investigate VitD outliers
vitd_outliers = df.loc[:, ['VitD_levels', 'vitD_supp']]
vitd_outliers['VitD_levels_outlier'] = df_outliers.VitD_levels
vitd_outliers['vitD_supp_outlier'] = df_outliers.vitD_supp
vitd_outliers = vitd_outliers.groupby(['VitD_levels_outlier', 'vitD_supp_outlier'], as_index=False). \
    mean().sort_values(['VitD_levels_outlier', 'vitD_supp_outlier'], ascending=False)
print(f'Vitamin D Outliers\n{vitd_outliers.to_string()}')
input('Enter to continue...\n') if INTERACTIVE else print()

# Investigate Doc_visits outliers
doc_visits_outliers = medical_clean.loc[df_outliers.Doc_visits, ['Doc_visits', 'Customer_id']]
doc_visits_outliers.columns = ['Doc_visits', 'count']
doc_visits_outliers = doc_visits_outliers.groupby('Doc_visits', as_index=False).count()
print(f'Doc_visits Outliers:\n{doc_visits_outliers.to_string()}')
input('Enter to continue...\n') if INTERACTIVE else print()

# Investigate Full_meals_eaten outliers
full_meals_eaten_outliers = medical_clean.loc[df_outliers.Full_meals_eaten, ['Full_meals_eaten', 'Customer_id']]
full_meals_eaten_outliers.columns = ['Full_meals_eaten', 'count']
full_meals_eaten_outliers = full_meals_eaten_outliers.groupby('Full_meals_eaten', as_index=False).count()
print(f'Full_meals_eaten Outliers:\n{full_meals_eaten_outliers.to_string()}')
input('Enter to continue...\n') if INTERACTIVE else print()

print('\n#---------------------------#\n'
      '| EXPLORATORY DATA ANALYSIS |\n'
      '#---------------------------#\n')
# Univariate analysis
# Quantitative variables
print(f'Quantitative Variables:\n{df.describe().T.to_string()}')
input('Enter to continue...\n') if INTERACTIVE else print()

# Qualitative variables
cat_cols = [column for column in columns if df[column].dtype not in ['float64', 'int64']]
print(f'Qualitative Variables:\n{df.loc[:, cat_cols].describe().T.to_string()}\n\nUnique Values:')
for col in cat_cols:
    print(f'{col}: {df.loc[:, col].unique()}')
input('Enter to continue...\n') if INTERACTIVE else print()

# Visual of findings
if SHOW_PLOTS:
    s, e = 0, 6
    grid = []
    for i in range(int(np.ceil(df.shape[1] / e))):
        grid.append([x for x in df.columns.values[s:e]])
        s = e
        e += 6
    fig, ax = plt.subplots(len(grid), 6, figsize=(15, 8))
    fig.set_tight_layout(True)
    fig.suptitle('Univariate Analysis')
    for r, row in enumerate(grid):
        for c, col in enumerate(row):
            if df[col].dtype in ('float64', 'int64'):
                ax[r, c].hist(df[col])
                ax[r, c].vlines(df[col].median(), 0, 4000, color='black', label='median')
                ax[r, c].vlines(df[col].mean(), 0, 4000, color='red', linestyles='--', label='mean')
            else:
                df_grp = df.groupby(col).count()
                if col == 'Initial_admin':
                    df_grp.index = df_grp.index.str.replace(' Admission', '')
                ax[r, c].bar(df_grp.index, df_grp.values[:, 0])
                plt.setp(ax[r, c].get_xticklabels(), rotation=20, ha='right', rotation_mode='anchor')
            ax[r, c].set_xlabel(col)
    plt.savefig('./output2/univariate_analysis.png')
    print('Close the plot window to continue...\n')
    plt.show()
    plt.close()

# Bivariate analysis
# Visual of findings
if SHOW_PLOTS:
    s, e = 0, 6
    grid = []
    for i in range(int(np.ceil(len(df.columns) / e))):
        grid.append([x for x in df.columns[1:].values[s:e]])
        s = e
        e += 6
    fig, ax = plt.subplots(len(grid), 6, sharey='row', figsize=(15, 8))
    fig.set_tight_layout(True)
    fig.suptitle('Bivariate Analysis')
    for r, row in enumerate(grid):
        for c, col in enumerate(row):
            if df[col].dtype in ('float64', 'int64'):
                readmis_yes = df.loc[df.ReAdmis.eq('Yes'), col].values
                readmis_no = df.loc[df.ReAdmis.eq('No'), col].values
                ax[r, c].boxplot([readmis_yes, readmis_no], labels=['Yes', 'No'], vert=False, sym='', widths=0.5)
                ax[r, c].set_xlabel(col)
                if c == 0:
                    ax[r, c].set_ylabel('ReAdmis')
            else:  # True
                readmis_yes = df.loc[df.ReAdmis.eq('Yes'), ['ReAdmis', col]].groupby(col).count()['ReAdmis'].values
                readmis_no = df.loc[df.ReAdmis.eq('No'), ['ReAdmis', col]].groupby(col).count()['ReAdmis'].values
                labels = df.loc[:, col].str.replace(' Admission', '').unique()
                x = np.arange(len(labels))
                width = 0.35
                bar1 = ax[r, c].bar(x - width/2, readmis_yes, width, label='ReAdmis=Yes')
                bar2 = ax[r, c].bar(x + width/2, readmis_no, width, label='ReAdmis=No')
                ax[r, c].set_xlabel(col)
                ax[r, c].set_xticks(x, labels)
                if c == 0:
                    ax[r, c].set_ylabel('count')
                plt.setp(ax[r, c].get_xticklabels(), rotation=20, ha='right', rotation_mode='anchor')
    blue = mpatches.Patch(color='#1f77b4', label='Yes')
    orange = mpatches.Patch(color='#ff7f0e', label='No')
    ax[4, 5].legend(title='ReAdmis', handles=[blue, orange], loc='center')
    plt.savefig('./output2/bivariate_analysis.png')
    print('Close the plot window to continue...\n')
    plt.show()
    plt.close()

print('\n#----------------#\n'
      '| DATA WRANGLING |\n'
      '#----------------#\n')
# Recode categorical variables
num_cols = [col for col in df.columns if df[col].dtype in ('int64', 'float64')]
df = df.replace({'Yes': 1, 'No': 0})
cat_cols = [col for col in df.columns if df[col].dtype not in ('int64', 'float64')]
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Fix column names
df.columns = [col.replace(' ', '_') for col in df.columns]
print(f'Recoded data:\n{df.head().to_string()}')
input('Enter to continue...\n') if INTERACTIVE else print()

# Scale numeric variables (not including 1/0)
scaler = StandardScaler()
df.loc[:, num_cols] = scaler.fit_transform(df.loc[:, num_cols])
print(f'Scaled data:\n{df.head().to_string()}')
input('Enter to continue...\n') if INTERACTIVE else print()

# Collinearity check
x = df.loc[:, df.columns[1:]]
vif_data = pd.DataFrame()
vif_data['feature'] = x.columns
vif_data['vif'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
print(f'Variance Inflation Factor:\n{vif_data.to_string()}')
input('Enter to continue...\n') if INTERACTIVE else print()

# Remove high VIFs
x.pop('Initial_days')
x.pop('Additional_charges')

# Collinearity check again
vif_data = pd.DataFrame()
vif_data['feature'] = x.columns
vif_data['vif'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
print(f'Variance Inflation Factor:\n{vif_data.to_string()}')
input('Enter to continue...\n') if INTERACTIVE else print()

# Save prepared data set
df.to_csv('./output2/prepared_data_set.csv')


print('\n#----------------------#\n'
      '| GROSS LOGISTIC MODEL |\n'
      '#----------------------#\n')
formula = f'ReAdmis ~ {" + ".join([var for var in x])}'
model1 = logit(formula, df).fit()
print(f'Gross Logistic Model:\n{model1.summary()}\n')
input('Enter to continue...\n') if INTERACTIVE else print()


print('\n#-------------------#\n'
      '| FEATURE SELECTION |\n'
      '#-------------------#\n')


def reduce_model(model: logit, data: pd.DataFrame) -> logit:
    features = model.params.index.to_list()  # Get the list of features
    features.pop(0)  # Remove the intercept from the list of features
    highest_p = model.pvalues.argmax() - 1  # Get the index of the highest p-value
    print(f'Removing {features[highest_p]} with a p-value of {float(model.pvalues[highest_p + 1])}')
    features.pop(highest_p)  # Remove the item with the highest p-value
    formula = f'ReAdmis ~ {" + ".join(features)}'  # Rewrite the formula
    model = logit(formula=formula, data=data).fit()  # Rebuild the model
    return model


model2 = model1
while model2.pvalues.max() >= 0.05:
    model2 = reduce_model(model2, df)
input('Enter to continue...\n') if INTERACTIVE else print()


print('\n#----------------#\n'
      '| MODEL ANALYSIS |\n'
      '#----------------#\n')
# Model Assumptions
print(f'Verifying model assumptions:\n\n'
      f'Model assumption 1: binary response variable:\n'
      f'ReAdmis unique values: {df.ReAdmis.nunique()}')
input('Enter to continue...\n') if INTERACTIVE else print()

print(f'Model assumption 2: independent observations:\n'
      f'Data contains only unique patients')
input('Enter to continue...\n') if INTERACTIVE else print()

print(f'Model assumption 3: no multicollinearity')
x = df.loc[:, model2.params.index.to_list()[1:]]
vif_data = pd.DataFrame()
vif_data['feature'] = x.columns
vif_data['vif'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
print(f'Variance Inflation Factor:\n{vif_data.to_string()}')
input('Enter to continue...\n') if INTERACTIVE else print()

print(f'Model assumption 4: no outliers\n'
      f'Several outliers were detected and kept during data cleaning')
input('Enter to continue...\n') if INTERACTIVE else print()

print(f'Model assumption 5: linearity')
# Box-Tidwell test
# https://towardsdatascience.com/assumptions-of-logistic-regression-clearly-explained-44d85a22b290#:~:text=Assumption%202%20%E2%80%94%20Linearity%20of%20independent%20variables%20and%20log%2Dodds
test_data = pd.DataFrame(medical_clean.loc[:, ['Children', 'TotalCharge']])
test_data.loc[:, 'Children_x_Log_Children'] = test_data.Children * np.log1p(test_data.Children)
test_data.loc[:, 'TotalCharge_x_Log_TotalCharge'] = test_data.TotalCharge * np.log1p(test_data.TotalCharge)
test_data = add_constant(test_data, prepend=False)
bt_results = GLM(df.ReAdmis, test_data, family=Binomial()).fit()
print(bt_results.summary())
print('No linear relationship exists between ReAdmis and Children_x_Log_Children')
params = model2.params.index.to_list()
params.pop(params.index('Intercept'))
params.pop(params.index('Children'))
formula = f'ReAdmis ~ ' + ' + '.join(params)
model2 = logit(formula, df).fit()
print('The Children variable was removed from the reduced model')
input('Enter to continue...\n') if INTERACTIVE else print()

print('Model assumption 6: sufficient sample size')
min_sample = 10
n_vars = len(params)
min_prob = df[df.ReAdmis.eq(1)].shape[0] / df.shape[0]
print(f'Min sample per group: {min_sample}, Variables: {n_vars}, Min Probability: {min_prob}')
print(f'Minimum sample size: {(min_sample * n_vars) / min_prob}')
print(f'Actual sample size: {df.shape[0]}')
input('Enter to continue...\n') if INTERACTIVE else print()

# Confusion Matrix
index = ['actual false', 'actual true']
columns = ['predicted false', 'predicted true']
conf_matrix1 = pd.DataFrame(model1.pred_table(), index=index, columns=columns)
conf_matrix2 = pd.DataFrame(model2.pred_table(), index=index, columns=columns)
print(f'Gross model confusion matrix:\n{conf_matrix1.to_string()}\n')
print(f'Reduced model confusion matrix:\n{conf_matrix2.to_string()}')
input('Enter to continue...\n') if INTERACTIVE else print()

# Gross Model Metrics
tn1 = conf_matrix1.loc['actual false', 'predicted false']
fp1 = conf_matrix1.loc['actual false', 'predicted true']
fn1 = conf_matrix1.loc['actual true', 'predicted false']
tp1 = conf_matrix1.loc['actual true', 'predicted true']
accuracy1 = (tn1 + tp1) / (tn1 + fn1 + fp1 + tp1)
sensitivity1 = tp1 / (fn1 + tp1)
specificity1 = tn1 / (tn1 + fp1)
print(f'Gross Model Metrics:\n'
      f'\taccuracy: {accuracy1}\n'
      f'\tsensitivity: {sensitivity1}\n'
      f'\tspecificity: {specificity1}\n')

# Reduced Model Metrics
tn2 = conf_matrix2.loc['actual false', 'predicted false']
fp2 = conf_matrix2.loc['actual false', 'predicted true']
fn2 = conf_matrix2.loc['actual true', 'predicted false']
tp2 = conf_matrix2.loc['actual true', 'predicted true']
accuracy2 = (tp2 + tn2) / (tn2 + fn2 + fp2 + tp2)
sensitivity2 = tp2 / (fn2 + tp2)
specificity2 = tn2 / (tn2 + fp2)
print(f'Reduced Model Metrics:\n'
      f'\taccuracy: {accuracy2}\n'
      f'\tsensitivity: {sensitivity2}\n'
      f'\tspecificity: {specificity2}')
input('Enter to continue...\n') if INTERACTIVE else print()

# Make data
TotalCharge = np.arange(medical_clean.TotalCharge.min(),
                        medical_clean.TotalCharge.max() + 1,
                        medical_clean.TotalCharge.std() / 2)
HighBlood = np.array([1, 0], int)
Stroke = np.array([1, 0], int)
Arthritis = np.array([1, 0], int)
Diabetes = np.array([1, 0], int)
Hyperlipidemia = np.array([1, 0], int)
BackPain = np.array([1, 0], int)
Anxiety = np.array([1, 0], int)
Allergic_rhinitis = np.array([1, 0], int)
Reflux_esophagitis = np.array([1, 0], int)
Asthma = np.array([1, 0], int)
Initial_admin_Emergency_Admission = np.array([1, 0], int)
Initial_admin_Observation_Admission = np.array([1, 0], int)
Complication_risk_Low = np.array([1, 0], int)
Complication_risk_Medium = np.array([1, 0], int)
Services_CT_Scan = np.array([1, 0], int)
Services_MRI = np.array([1, 0], int)
p = product(TotalCharge, HighBlood, Stroke, Arthritis, Diabetes, Hyperlipidemia, BackPain, Anxiety, Allergic_rhinitis,
            Reflux_esophagitis, Asthma, Initial_admin_Emergency_Admission, Initial_admin_Observation_Admission,
            Complication_risk_Low, Complication_risk_Medium, Services_CT_Scan, Services_MRI)
prediction_data = pd.DataFrame(p, columns=model2.params.index.to_list()[1:])
# Scale Initial_days
scaler = StandardScaler()
prediction_data.loc[:, ['TotalCharge', 'HighBlood_']] = \
    scaler.fit_transform(prediction_data.loc[:, ['TotalCharge', 'HighBlood']])
prediction_data.drop('HighBlood_', axis=1, inplace=True)
# Make Predictions
prediction_data['ReAdmis'] = model2.predict(prediction_data)
# Unscale Initial_days
prediction_data.loc[:, ['TotalCharge', 'HighBlood_']] = \
    scaler.inverse_transform(prediction_data.loc[:, ['TotalCharge', 'HighBlood']])
prediction_data.drop('HighBlood_', axis=1, inplace=True)
print(f'Predictions:\n{prediction_data.describe().T.to_string()}')
input('Enter to continue...\n') if INTERACTIVE else print()

# Plot Predictions
if SHOW_PLOTS:
    sns.regplot(x=medical_clean.TotalCharge, y=df.ReAdmis, logistic=True, label='observations')
    sns.scatterplot(x=prediction_data.TotalCharge, y=np.round(prediction_data.ReAdmis), color='red', label='predictions')
    plt.title('Gross Logistic Regression')
    plt.legend()
    plt.savefig('./output2/predictions.png')
    print('Close the plot window to continue...')
    plt.show()
    plt.close()

# Calculate and plot odds ratio
prediction_data['odds_ratio'] = prediction_data.ReAdmis / (1 - prediction_data.ReAdmis)
x = [' + '.join(x)]
if SHOW_PLOTS:
    sns.lineplot(x='TotalCharge', y='odds_ratio', data=prediction_data)
    plt.axhline(y=1, linestyle='dotted')
    plt.yscale('log')
    plt.title('TotalCharge vs ReAdmis odds ratio')
    plt.savefig('./output2/odds_ratio.png')
    print('Close the plot window to continue...')
    plt.show()
    plt.close()


print('\n#------------------------#\n'
      '| REDUCED LOGISTIC MODEL |\n'
      '#------------------------#\n')
# Reduced model
print(model2.summary())
