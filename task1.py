import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols
from statsmodels.api import qqplot
from itertools import product


SHOW_PLOTS = True

# Create output1 directory
os.makedirs('./output1', exist_ok=True)

# Load data
medical_clean = pd.read_csv(r"C:\Users\user\OneDrive\Documents\Education\Western Govenors University"
                            r"\MS - Data Analytics\D208 - Predictive Modeling\medical_clean.csv")

# Select dependent variable (continuous)
# and all potential independent variables (at least one categorical)
# Dependent = Initial_days
# Independent: Everything that could be relevant to start
columns = ['Initial_days', 'Lat', 'Lng', 'Children', 'Age', 'Income', 'VitD_levels', 'Doc_visits', 'Full_meals_eaten',
           'vitD_supp', 'TotalCharge', 'Additional_charges', 'Soft_drink', 'HighBlood', 'Stroke', 'Overweight',
           'Arthritis', 'Diabetes', 'Hyperlipidemia', 'BackPain', 'Anxiety', 'Allergic_rhinitis', 'Reflux_esophagitis',
           'Asthma', 'Area', 'Marital', 'Gender', 'Initial_admin', 'Complication_risk', 'Services']
df = medical_clean[columns]

print('\n#---------------#\n'
      '| DATA CLEANING |\n'
      '#---------------#\n')
# Duplicate detection
print(f'Duplication check:\n\tNumber of duplicated rows: {df.duplicated().sum()}')
input('Enter to continue...\n')

# Null Detection/Handling
print(f'Null detection:\n{df.isna().sum()}')
input('Enter to continue...\n')

# Outlier Detection/Handling
num_cols = [column for column in columns if df[column].dtype in ['float64', 'int64']]
df_num = df[num_cols]
df_zscores = df_num.apply(stats.zscore)
df_outliers = df_zscores.apply(lambda x: (x > 3) | (x < -3))
print(f'Outlier counts:\n{df_outliers.sum().to_string()}')
input('Enter to continue...\n')

# Investigate Location outliers
location_outliers = medical_clean.loc[df_outliers.Lat | df_outliers.Lng, ['State', 'Lat', 'Lng']]
location_outliers.columns = ['State', 'Lat_mean', 'Lng_mean']
location_outlier_groups = location_outliers.groupby("State", as_index=False).mean()
print(f'Location Outliers:\n{location_outlier_groups.to_string()}')
input('Enter to continue...\n')
if SHOW_PLOTS:
    yes = df.loc[df_outliers.Lat | df_outliers.Lng, ['Lat', 'Lng']]
    no = df.loc[~df_outliers.Lat & ~df_outliers.Lng, ['Lat', 'Lng']]
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle('Location Outliers')
    ax.scatter(x=yes.Lng, y=yes.Lat, c='red', s=1, label='Yes')
    ax.scatter(x=no.Lng, y=no.Lat, c='blue', s=.1, label='No')
    ax.set_xlabel('Lng')
    ax.set_ylabel('Lat')
    ax.legend(title='Outlier')
    plt.savefig('./output1/location_outliers.png')
    print('Close the plot window to continue...')
    plt.show()
    plt.close()

# Investigate Children outliers
children_outliers = medical_clean.loc[df_outliers.Children, ['Children', 'Customer_id']]
children_outliers.columns = ['Children', 'count']
children_outliers = children_outliers.groupby("Children", as_index=False).count()
print(f'Children Outliers:\n{children_outliers.to_string()}')
input('Enter to continue...\n')

# Investigate Income outliers
income_outliers = medical_clean.loc[df_outliers.Income, ['Job', 'Income']]
income_outliers.columns = ['Job', 'Income_mean']
income_outliers = income_outliers.groupby("Job", as_index=False).mean(). \
    sort_values("Income_mean", ascending=False)
print(f'Income Outliers:\n{income_outliers.iloc[:10].to_string()}')
input('Enter to continue...\n')

# Investigate VitD outliers
vitd_outliers = df.loc[:, ['VitD_levels', 'vitD_supp']]
vitd_outliers['VitD_levels_outlier'] = df_outliers.VitD_levels
vitd_outliers['vitD_supp_outlier'] = df_outliers.vitD_supp
vitd_outliers = vitd_outliers.groupby(['VitD_levels_outlier', 'vitD_supp_outlier'], as_index=False). \
    mean().sort_values(['VitD_levels_outlier', 'vitD_supp_outlier'], ascending=False)
print(f'Vitamin D Outliers\n{vitd_outliers.to_string()}')
input('Enter to continue...\n')

# Investigate Doc_visits outliers
doc_visits_outliers = medical_clean.loc[df_outliers.Doc_visits, ['Doc_visits', 'Customer_id']]
doc_visits_outliers.columns = ['Doc_visits', 'count']
doc_visits_outliers = doc_visits_outliers.groupby('Doc_visits', as_index=False).count()
print(f'Doc_visits Outliers:\n{doc_visits_outliers.to_string()}')
input('Enter to continue...\n')

# Investigate Full_meals_eaten outliers
full_meals_eaten_outliers = medical_clean.loc[df_outliers.Full_meals_eaten, ['Full_meals_eaten', 'Customer_id']]
full_meals_eaten_outliers.columns = ['Full_meals_eaten', 'count']
full_meals_eaten_outliers = full_meals_eaten_outliers.groupby('Full_meals_eaten', as_index=False).count()
print(f'Full_meals_eaten Outliers:\n{full_meals_eaten_outliers.to_string()}')
input('Enter to continue...\n')

print('\n#---------------------------#\n'
      '| EXPLORATORY DATA ANALYSIS |\n'
      '#---------------------------#\n')
# Univariate analysis
# Quantitative variables
print(f'Quantitative Variables:\n{df.describe().T.to_string()}')
input('Enter to continue...\n')

# Qualitative variables
cat_cols = [column for column in columns if df[column].dtype not in ['float64', 'int64']]
print(f'Qualitative Variables:\n{df.loc[:, cat_cols].describe().T.to_string()}\n\nUnique Values:')
for col in cat_cols:
    print(f'{col}: {df.loc[:, col].unique()}')
input('Enter to continue...\n')

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
    plt.savefig('./output1/univariate_analysis.png')
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
    fig, ax = plt.subplots(len(grid), 6, sharey='all', figsize=(15, 8))
    fig.set_tight_layout(True)
    fig.suptitle('Bivariate Analysis')
    for r, row in enumerate(grid):
        for c, col in enumerate(row):
            if df[col].dtype in ('float64', 'int64'):
                ax[r, c].scatter(df[col], df.Initial_days, marker='.', alpha=.25)
            else:
                df_grp = {}
                for grp in df.loc[:, col].unique():
                    df_grp[grp] = df.loc[df[col].eq(grp), 'Initial_days'].values
                labels = [x.replace(' Admission', '') for x in df_grp.keys()]
                ax[r, c].boxplot(df_grp.values(), labels=labels, widths=0.5, patch_artist=True,
                                 medianprops={'color': 'white', 'linewidth': 0.5},
                                 boxprops={'facecolor': 'C0', 'edgecolor': 'white', 'linewidth': 0.5},
                                 whiskerprops={'color': 'C0', 'linewidth': 1.5},
                                 capprops={'color': 'C0', 'linewidth': 1.5})
                plt.setp(ax[r, c].get_xticklabels(), rotation=20, ha='right', rotation_mode='anchor')
            ax[r, c].set_xlabel(col)
            if c == 0:
                ax[r, c].set_ylabel('Initial_days')
    plt.savefig('./output1/bivariate_analysis.png')
    print('Close the plot window to continue...\n')
    plt.show()
    plt.close()

print('\n#----------------#\n'
      '| DATA WRANGLING |\n'
      '#----------------#\n')
# Recode categorical variables
codes = {'Yes': 1, 'No': 0}
df = df.replace(codes)
df = pd.get_dummies(df, columns=['Area', 'Marital', 'Gender', 'Initial_admin', 'Services', 'Complication_risk'],
                    drop_first=True)

# Fix column names
columns = [col.replace(' ', '_') for col in df.columns]
df.columns = columns
print(f'Recoded data:\n{df.head().to_string()}')
input('Enter to continue...\n')

# Scale numeric variables (not including 1/0 columns)
scaler = StandardScaler()
df.loc[:, df_num.columns] = scaler.fit_transform(df_num)
print(f'Scaled Data:\n{df.head().to_string()}')
input('Enter to continue...\n')

# Collinearity check
x = df.loc[:, df.columns[1:]]
vif_data = pd.DataFrame()
vif_data['feature'] = x.columns
vif_data['vif'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
print(f'Variance Influence Factor:\n{vif_data.to_string()}')
input('Enter to continue...\n')

# Drop variable(s) with high VIF
df.drop('Additional_charges', axis=1, inplace=True)

# Check Collinearity again
x = df.loc[:, df.columns[1:]]
vif_data = pd.DataFrame()
vif_data['feature'] = x.columns
vif_data['vif'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
print(f'Updated Variance Influence Factor:\n{vif_data.to_string()}')
input('Enter to continue...\n')

# Prepared data set
df.to_csv('./output1/prepared_data_set.csv')

print('\n#--------------------#\n'
      '| GROSS LINEAR MODEL |\n'
      '#--------------------#\n')
columns = df.columns.values.tolist()
columns.pop(0)
formula = f'Initial_days ~ {" + ".join([col for col in columns])}'
model1 = ols(formula, data=df).fit()
print(model1.summary())
input('Enter to continue...\n')

print('\n#-------------------#\n'
      '| FEATURE SELECTION |\n'
      '#-------------------#\n')


def reduce_model(model: ols, data: pd.DataFrame) -> ols:
    features = model.params.index.to_list()  # Get the list of features
    features.pop(0)  # Remove the intercept from the list of features
    highest_p = model.pvalues.argmax() - 1  # Get the index of the highest p-value
    print(f'Removing {features[highest_p]} with a p-value of {float(model.pvalues[highest_p + 1])}')
    features.pop(highest_p)  # Remove the item with the highest p-value
    formula = f'Initial_days ~ {" + ".join(features)}'  # Rewrite the formula
    model = ols(formula=formula, data=data).fit()  # Rebuild the model
    if any(model.pvalues.round(2)) > 0.05:  # Recursively reduce the model until no insignificant features remain
        model = reduce_model(model, data)
    return model


model2 = reduce_model(model1, df)
input('Enter to continue...\n')

print('\n#----------------------#\n'
      '| REDUCED LINEAR MODEL |\n'
      '#----------------------#\n')
print(model2.summary())
input('Enter to continue...\n')

print('\n#----------------#\n'
      '| MODEL ANALYSIS |\n'
      '#----------------#\n')
# Residuals Gross vs Reduced
if SHOW_PLOTS:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle('Residuals Distributions')
    ax1.set_title('Gross Model')
    ax1.hist(model1.resid)
    ax2.set_title('Reduced Model')
    ax2.hist(model2.resid)
    plt.savefig('./output1/residuals_distributions_compare.png')
    print('Close the plot window to continue...\n')
    plt.show()
    plt.close()

# Calculate residual standard error from Mean Squared Error (mse)
print(f'Residual Standard Error:\n'
      f'\tGross Model: {np.sqrt(model1.mse_resid)}\n'
      f'\tReduced Model: {np.sqrt(model2.mse_resid)}\n')
input('Enter to continue...\n')

# Check linearity
df_model = df.loc[:, ['Initial_days', 'TotalCharge', 'HighBlood', 'Arthritis', 'Diabetes', 'Hyperlipidemia',
                      'BackPain', 'Anxiety', 'Allergic_rhinitis', 'Reflux_esophagitis',
                      'Initial_admin_Emergency_Admission', 'Complication_risk_Low', 'Complication_risk_Medium']]
if SHOW_PLOTS:
    corr = df_model.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.set_tight_layout(True)
    fig.suptitle('Correlation Matrix')
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='Spectral')
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(corr.shape[1]), labels=corr.columns)
    ax.set_yticks(np.arange(corr.shape[0]), labels=corr.index)
    plt.setp(ax.get_xticklabels(), rotation=25, ha='right', rotation_mode='anchor')
    for i, index in enumerate(corr.index):
        for c, col in enumerate(corr.columns):
            text = ax.text(c, i, '{:.4f}'.format(corr.loc[index, col]).replace('0.', '.').
                           replace('1.0000', ''), ha='center', va='center', size=7,
                           color='w' if corr.loc[index, col] > 0.5 else 'black')
    print('Close the plot window to continue...\n')
    plt.savefig('./output1/correlation_plot.png')
    plt.show()
    plt.close()

# Check Multicollinearity
x = df_model.loc[:, df_model.columns[1:]]
vif_data = pd.DataFrame()
vif_data['feature'] = x.columns
vif_data['vif'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
print(f'Variance Influence Factor:\n{vif_data.to_string()}')
input('Enter to continue...\n')

# Check Independence of Observations and Homosceasticity
if SHOW_PLOTS:
    sns.regplot(x=model2.fittedvalues, y=model2.resid, lowess=True, line_kws={'color': 'red'})
    plt.title('Residuals vs. fitted')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.savefig('./output1/residuals_vs_fitted.png')
    print('Close the plot window to continue...\n')
    plt.show()
    plt.close()

# Check Normality of residuals
if SHOW_PLOTS:
    qqplot(data=model2.resid, fit=True, line='45')
    plt.title('Q-Q plot')
    plt.savefig('./output1/qqplot.png')
    print('Close the plot window to continue...\n')
    plt.show()
    plt.close()

# Scale-location plot
if SHOW_PLOTS:
    model2_norm_resid = model2.get_influence().resid_studentized_internal
    model2_norm_resid_abs_sqrt = np.sqrt(np.abs(model2_norm_resid))
    sns.regplot(x=model2.fittedvalues, y=model2_norm_resid_abs_sqrt, ci=None, lowess=True, line_kws={'color': 'red'})
    plt.title('Scale-location plot')
    plt.xlabel('Fitted values')
    plt.ylabel('|Standardized Residuals|**(1/2)')
    plt.savefig('./output1/scale-location.png')
    print('Close the plot window to continue...\n')
    plt.show()
    plt.close()

# Make predictions
# Create prediction data
total_charge = np.arange(2000, 11000, 500)
high_blood = np.array([0, 1], int)
arthritis = np.array([0, 1], int)
diabetes = np.array([0, 1], int)
hyperlipidemia = np.array([0, 1], int)
back_pain = np.array([0, 1], int)
anxiety = np.array([0, 1], int)
allergic_rhinitis = np.array([0, 1], int)
reflux_esophagitis = np.array([0, 1], int)
emergency_admission = np.array([0, 1], int)
low_risk = np.array([0, 1], int)
medium_risk = np.array([0, 1], int)
p = product(total_charge, high_blood, arthritis, diabetes, hyperlipidemia, back_pain, anxiety, allergic_rhinitis,
            reflux_esophagitis, emergency_admission, low_risk, medium_risk)
explanatory_data = pd.DataFrame(p, columns=df_model.columns[1:])
# Scale numeric part of data
scaler = StandardScaler()
scaler.fit(explanatory_data.loc[:, ['TotalCharge', 'HighBlood']])
explanatory_data.loc[:, ['TotalCharge', 'HighBlood_']] = \
    scaler.transform(explanatory_data.loc[:, ['TotalCharge', 'HighBlood']])
explanatory_data.drop('HighBlood_', axis=1, inplace=True)
# Make predictions
prediction_data = explanatory_data.assign(Initial_days=model2.predict(explanatory_data))
# Unscale data for interpretation
scaler = StandardScaler()
scaler.fit(medical_clean.loc[:, ['TotalCharge', 'Initial_days']])
prediction_data.loc[:, ['TotalCharge', 'Initial_days']] = \
    scaler.inverse_transform(prediction_data.loc[:, ['TotalCharge', 'Initial_days']])
df.loc[:, ['TotalCharge', 'Initial_days']] = scaler.inverse_transform(df.loc[:, ['TotalCharge', 'Initial_days']])
print(f'Predictions:\n{prediction_data.describe().T.to_string()}')
# Plot predictions
sns.scatterplot(x=df.TotalCharge, y=df.Initial_days, hue=df.Initial_admin_Emergency_Admission, alpha=.25)
sns.scatterplot(x=prediction_data.TotalCharge, y=prediction_data.Initial_days,
                hue=prediction_data.Initial_admin_Emergency_Admission, legend=False, marker='s', alpha=.25)
plt.title('Predicted vs. Actual')
plt.savefig('./output1/predicted_vs_actual.png')
print('Close the plot window to end the program...\n')
plt.show()
plt.close()
