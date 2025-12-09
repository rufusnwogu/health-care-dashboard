# importing the libraries needed for the project
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# Construct the full path to the CSV file
data_file_path = "hospital_readmissions.csv"

# Load the dataset
data = pd.read_csv(data_file_path)

# Preview dataset
print("\n First five rows of the dataset:")
print(data.head())


# Basic info
print("\n Dataset Info:")
print(data.info())


# Dataset shape
print(f"\n Dataset shape: {data.shape}")


# Check for missing data
print("\n Missing values per column:")
print(data.isnull().sum())


# Check data types
print("\n Data types:")
print(data.dtypes)


# ### Feature Engineering and Data Cleaning
# --------------------------------------
# STEP 2: FEATURE ENGINEERING
# --------------------------------------

# Convert age ranges to numeric midpoints
def age_to_mid(age_range):
    """Convert an age range like '[70-80)' to a numeric midpoint (e.g., 75)."""
    low, high = age_range.strip('[]()').split('-')
    return int(round((int(low) + int(high)) / 2))

# Apply transformation
data['age_mid'] = data['age'].apply(age_to_mid)


# Handle target variable ('readmitted')
data['readmitted'] = data['readmitted'].map({'no': 0, 'yes': 1})

# Create new engineered features
data['total_visits'] = data['n_outpatient'] + data['n_inpatient'] + data['n_emergency']
data['procedure_intensity'] = data['n_lab_procedures'] / (data['time_in_hospital'] + 1)
data['medication_per_day'] = data['n_medications'] / (data['time_in_hospital'] + 1)

data.head()

# --------------------------------------
# STEP 4: CHECK CLEAN DATA
# --------------------------------------
print("\n🧹 Missing values per column (after cleaning):")
print(data.isnull().sum())

print("\n Sample after feature engineering:")
print(data.head())


# ### Exploratory Data Analysis


# #### Basic Descriptive Statistics
print("Dataset Information:")
print(data.info())

print("\n Descriptive Statistics:")
print(data.describe().T)

categorical_cols = ['medical_specialty', 'diag_1', 'diag_2', 'diag_3']
for col in categorical_cols:
    print(f"\nUnique values in {col}: {data[col].nunique()}")


# #### Correlations


# ##### Correlation of Numerical Features
# Select only numerical columns for correlation analysis
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = data[numerical_cols].corr()

# Visualize the correlation matrix using a heatmap
fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto",
                title='Correlation Matrix of Numerical Features')
fig.update_layout(xaxis_title="Features", yaxis_title="Features")
fig.show()

print("\nCorrelation Matrix (Numerical Features):\n")
print(correlation_matrix)


# Select only numerical columns for correlation analysis, including 'readmitted'
numerical_cols_with_target = data.select_dtypes(include=['int64', 'float64']).columns

# Calculate correlation with 'readmitted'
correlation_with_readmitted = data[numerical_cols_with_target].corr()['readmitted'].sort_values(ascending=False)

print("\nCorrelation with 'readmitted' column:\n")
print(correlation_with_readmitted)

# Visualize the correlation with 'readmitted'
fig = px.bar(correlation_with_readmitted.drop('readmitted'),
             x=correlation_with_readmitted.drop('readmitted').index,
             y=correlation_with_readmitted.drop('readmitted').values,
             title='Correlation of Numerical Features with Readmitted',
             labels={'x':'Numerical Feature', 'y':'Correlation Coefficient'})
fig.update_xaxes(categoryorder='total descending')
fig.show()


# ##### Correlation of Categorical Features with 'readmitted'
categorical_features_for_corr = ['medical_specialty', 'diag_1', 'diag_2', 'diag_3']

for col in categorical_features_for_corr:
    # Calculate readmission rate for each category
    readmission_rate_by_category = data.groupby(col)['readmitted'].mean().sort_values(ascending=False).reset_index()

    # Get top 10 categories for visualization
    top_10_categories = readmission_rate_by_category.head(10)

    # Plotting the readmission rates for the top 10 categories
    fig = px.bar(top_10_categories,
                 x=col,
                 y='readmitted',
                 title=f'Readmission Rate by {col} (Top 10)',
                 labels={'readmitted': 'Readmission Rate', col: col},
                 color=col)
    fig.update_layout(xaxis_title=col, yaxis_title='Average Readmission Rate (0-1)')
    fig.show()

    print(f"\nReadmission rates for top categories in {col}:\n")
    print(top_10_categories.round(3))

# #### Variable Analysis
# ##### Univariate Analysis
print("\n--- Univariate Analysis ---\n")

# Distribution of Target Variable: readmitted
readmitted_counts = data['readmitted'].value_counts().reset_index()
readmitted_counts.columns = ['Readmitted', 'Count']
fig = px.bar(readmitted_counts, x='Readmitted', y='Count', title='Distribution of Readmitted Patients')
fig.update_layout(xaxis_title="Readmitted (0=No, 1=Yes)", yaxis_title="Number of Patients")
fig.show()

# Distribution of age_mid
fig = px.histogram(data, x='age_mid', title='Distribution of Patient Age (Midpoint)')
fig.show()

# Distribution of time_in_hospital
fig = px.histogram(data, x='time_in_hospital', title='Distribution of Time in Hospital (Days)')
fig.show()

# Distribution of n_medications
fig = px.histogram(data, x='n_medications', title='Distribution of Number of Medications')
fig.show()

# Distribution of medical_specialty (Top 10)
top_medical_specialties = data['medical_specialty'].value_counts().head(10).index
fig = px.bar(data[data['medical_specialty'].isin(top_medical_specialties)],
             x='medical_specialty', title='Top 10 Medical Specialties', color='medical_specialty')
fig.update_xaxes(categoryorder='total descending')
fig.show()

# Distribution of diag_1 (Top 10)
top_diag_1 = data['diag_1'].value_counts().head(10).index
fig = px.bar(data[data['diag_1'].isin(top_diag_1)],
             x='diag_1', title='Top 10 Primary Diagnoses', color='diag_1')
fig.update_xaxes(categoryorder='total descending')
fig.show()

# ##### Bivariate Analysis

print("\n--- Bivariate Analysis ---\n")

# Bivariate Analysis: Numerical Features vs. Readmitted

# Time in Hospital vs. Readmitted
fig = px.box(data, x='readmitted', y='time_in_hospital', title='Time in Hospital vs. Readmitted')
fig.update_layout(xaxis_title="Readmitted (0=No, 1=Yes)", yaxis_title="Time in Hospital (Days)")
fig.show()

# Age vs. Readmitted
fig = px.box(data, x='readmitted', y='age_mid', title='Age (Midpoint) vs. Readmitted')
fig.update_layout(xaxis_title="Readmitted (0=No, 1=Yes)", yaxis_title="Age (Midpoint)")
fig.show()

# Number of Inpatient Visits vs. Readmitted
# Using violin plot for better distribution insight given discrete nature and potential outliers
fig = px.violin(data, x='readmitted', y='n_inpatient', title='Number of Inpatient Visits vs. Readmitted')
fig.update_layout(xaxis_title="Readmitted (0=No, 1=Yes)", yaxis_title="Number of Inpatient Visits")
fig.show()

# Number of Medications vs. Readmitted
fig = px.box(data, x='readmitted', y='n_medications', title='Number of Medications vs. Readmitted')
fig.update_layout(xaxis_title="Readmitted (0=No, 1=Yes)", yaxis_title="Number of Medications")
fig.show()


# Bivariate Analysis: Categorical Features vs. Readmitted

# Medical Specialty vs. Readmission Rate (Top 10)
readmission_rate_by_specialty = data.groupby('medical_specialty')['readmitted'].mean().sort_values(ascending=False).reset_index()
top_10_specialties = readmission_rate_by_specialty.head(10)
fig = px.bar(top_10_specialties,
             x='medical_specialty',
             y='readmitted',
             title='Readmission Rate by Medical Specialty (Top 10)',
             labels={'readmitted': 'Average Readmission Rate', 'medical_specialty': 'Medical Specialty'},
             color='medical_specialty')
fig.update_xaxes(categoryorder='total descending')
fig.show()

# Diag_1 (Primary Diagnosis) vs. Readmission Rate (Top 10)
readmission_rate_by_diag1 = data.groupby('diag_1')['readmitted'].mean().sort_values(ascending=False).reset_index()
top_10_diag1 = readmission_rate_by_diag1.head(10)
fig = px.bar(top_10_diag1,
             x='diag_1',
             y='readmitted',
             title='Readmission Rate by Primary Diagnosis (Top 10)',
             labels={'readmitted': 'Average Readmission Rate', 'diag_1': 'Primary Diagnosis'},
             color='diag_1')
fig.update_xaxes(categoryorder='total descending')
fig.show()

# Change in Medication vs. Readmitted
fig = px.bar(data.groupby('change')['readmitted'].mean().reset_index(),
             x='change', y='readmitted', title='Readmission Rate by Change in Medication',
             labels={'change': 'Change in Medication', 'readmitted': 'Average Readmission Rate'})
fig.show()


# ###### Multi-Variate Analysis
# **Reasoning**:
# The subtask requires visualizing the interaction between 'age_mid', 'time_in_hospital', and 'readmitted' using a scatter plot. I will use `plotly.express` to create this plot, coloring the points by the 'readmitted' status.
# 
# 


fig = px.scatter(
    data,
    x='age_mid',
    y='time_in_hospital',
    color='readmitted',
    title='Age vs. Time in Hospital by Readmission Status',
    labels={'age_mid': 'Age Midpoint', 'time_in_hospital': 'Time in Hospital (Days)', 'readmitted': 'Readmitted (0=No, 1=Yes)'}
)
fig.show()


# **Reasoning**:
# The next step is to visualize the interaction between two categorical features (`medical_specialty`, `change`) and the target variable (`readmitted`) using a grouped bar chart to understand how these factors collectively influence readmission rates.
# 
# 


medical_specialty_change_readmitted = data.groupby(['medical_specialty', 'change'])['readmitted'].mean().reset_index()

fig = px.bar(
    medical_specialty_change_readmitted,
    x='medical_specialty',
    y='readmitted',
    color='change',
    barmode='group',
    title='Readmission Rate by Medical Specialty and Change in Medication',
    labels={
        'medical_specialty': 'Medical Specialty',
        'readmitted': 'Average Readmission Rate',
        'change': 'Change in Medication'
    }
)
fig.update_layout(xaxis_categoryorder='total descending')
fig.show()


# ### Saving the Final Data
# --------------------------------------
# STEP 3: CLEAN BINARY/ORDINAL COLUMNS
# --------------------------------------

# Columns that are truly binary (yes/no)
binary_yes_no_cols = ['change', 'diabetes_med']

for col in binary_yes_no_cols:
    data[col] = (
        data[col]
        .astype(str)
        .str.lower()
        .replace({'missing': np.nan, 'unknown': np.nan, '?': np.nan, 'nan': np.nan})
        .map({'yes': 1, 'no': 0})
        .fillna(0)
        .astype(int)
    )

# Columns with 'high', 'normal', 'no' (ordinal mapping)
ordinal_cols = ['glucose_test', 'A1Ctest']

for col in ordinal_cols:
    data[col] = (
        data[col]
        .astype(str)
        .str.lower()
        .replace({'missing': np.nan, 'unknown': np.nan, '?': np.nan, 'nan': np.nan})
        .map({'no': 0, 'normal': 1, 'high': 2})
        .fillna(0) # 'no' or not performed as 0
        .astype(int)
    )

data.head()


# --------------------------------------
# STEP 6: SAVE CLEANED DATA
# --------------------------------------

# Define a suitable output path for Colab's environment.
# The original path 'C:\Users\USER\Documents\Data Analytics Project\hospital_readmissions_cleaned.csv'
# is a Windows-specific path and would not work directly in Colab.
output_path = "hospital_readmissions_cleaned.csv"

data.to_csv(output_path, index=False)

print(f"\nCleaned dataset saved to: {output_path}")
