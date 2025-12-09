# ==============================================
# Streamlit Dashboard for Hospital Readmission
# ==============================================

# !pip install streamlit  # Run this in terminal: pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio
import os

# ----------------------------------------------
# Load Data
# ----------------------------------------------
import streamlit as st
import pandas as pd


@st.cache_data
def load_data(path="Dataset/hospital_readmissions_cleaned.csv"):
    """
    Load the cleaned hospital readmissions dataset.
    The cleaned version already includes:
    - Feature engineering (age_mid, total_visits, procedure_intensity, medication_per_day)
    - Encoded binary variables (change, diabetes_med)
    - Encoded ordinal variables (glucose_test, A1Ctest)
    - Encoded target variable (readmitted: 0=no, 1=yes)
    """
    df = pd.read_csv(path)
    return df

# Load the data
data = load_data()

# ----------------------------------------------
# Treatment Scoring Functions
# ----------------------------------------------
def score_a1c(prev, curr):
    if prev == "high" and curr == "normal":
        return 2
    if prev == "normal" and curr == "normal":
        return 1
    return 0

def score_glucose(prev, curr):
    if prev == "high" and curr == "normal":
        return 2
    if prev == "normal" and curr == "normal":
        return 1
    return 0

def score_readmission(prev, curr):
    if prev == "yes" and curr == "no":
        return 2
    if prev == "no" and curr == "no":
        return 1
    return 0

def score_utilization(prev_inp, prev_er, curr_inp, curr_er):
    inp_drop = curr_inp < prev_inp
    er_drop = curr_er < prev_er
    if inp_drop and er_drop:
        return 2
    if inp_drop or er_drop:
        return 1
    return 0

def score_medication(change, diabetes_med, improved):
    if change == "yes" and improved:
        return 2
    if diabetes_med == "yes" and not improved:
        return 1
    return 0

def compute_treatment_score(df):
    scores = []
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        a1c_score = score_a1c(prev["A1Ctest"], curr["A1Ctest"])
        glucose_score = score_glucose(prev["glucose_test"], curr["glucose_test"])
        readmit_score = score_readmission(prev["readmitted"], curr["readmitted"])
        util_score = score_utilization(prev["n_inpatient"], prev["n_emergency"],
                                       curr["n_inpatient"], curr["n_emergency"])
        improved = (curr["A1Ctest"] == "normal") or (curr["glucose_test"] == "normal")
        med_score = score_medication(curr["change"], curr["diabetes_med"], improved)

        total_score = a1c_score + glucose_score + readmit_score + util_score + med_score
        scores.append(total_score)

    scores.insert(0, None)  # first visit has no previous data
    df["treatment_score"] = scores
    return df

# ----------------------------------------------
# Streamlit Page Configuration
# ----------------------------------------------
st.set_page_config(
    page_title="Healthcare Outcomes Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------------------------
# Global Plot Theme (Light Modern)
# ----------------------------------------------
custom_colors = ['#00B2A9', '#10B981']  # Teal & soft green

pio.templates["custom_light"] = pio.templates["plotly_white"]
pio.templates["custom_light"].layout.update(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(color="#111827", family="Arial", size=14),
    title_font=dict(size=18, color="#1E3A8A", family="Arial Black"),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(0,0,0,0.1)",
        borderwidth=1,
        font=dict(color="#111827", size=12)
    ),
    xaxis=dict(
        showgrid=True,
        gridcolor="#E5E7EB",
        title_font=dict(size=14, color="#111827")
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="#E5E7EB",
        title_font=dict(size=14, color="#111827")
    ),
    colorway=custom_colors
)
pio.templates.default = "custom_light"

# Seaborn + Matplotlib Light Theme
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "axes.facecolor": "#FFFFFF",
    "figure.facecolor": "#FFFFFF",
    "axes.edgecolor": "#E5E7EB",
    "grid.color": "#E5E7EB",
    "text.color": "#111827"
})

# ----------------------------------------------
# Custom CSS Styling
# ----------------------------------------------
st.markdown("""
    <style>
        /* General app background */
        .stApp {
            background-color: #FFFFFF;
            color: #000000;
            font-family: "Arial", sans-serif;
        }

        /* Titles */
        h1, h2, h3 {
            color: #1E3A8A;
            font-weight: 700;
        }

        /* KPI Container (Shadow + Rounded Corners) */
        div[data-testid="stMetric"] {
            background-color: #FFFFFF;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.08);
            text-align: center;
        }

        /* KPI Labels and Values */
        [data-testid="stMetricLabel"] {
            color: #4B5563;
            font-size: 16px;
            font-weight: 600;
        }
        [data-testid="stMetricValue"] {
            color: #111827;
            font-size: 28px;
            font-weight: 700;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #F9FAFB;
            border-right: 1px solid #E5E7EB;
        }

        /* Sidebar headers and filters */
        .sidebar-content h2, .sidebar-content h3, .sidebar-content h4, .stSelectbox label, .stSlider label {
            color: #1E40AF;
        }

        /* Plot background */
        .plotly {
            background-color: white !important;
        }

        /* Divider */
        hr {
            border: 1px solid #E5E7EB;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------
# Dashboard Title
# ----------------------------------------------
st.title("Healthcare Outcomes Dashboard")
st.markdown("**Patient readmissions, chronic conditions, and treatment outcomes**")

# ----------------------------------------------
# Sidebar Navigation
# ----------------------------------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Main Dashboard", "Correlation Analysis", "Variable Analysis"],
    index=0
)

st.sidebar.markdown("---")

# ----------------------------------------------
# Sidebar Filters
# ----------------------------------------------
st.sidebar.header("🔍 Data Filters")

age_filter = st.sidebar.slider(
    "Select Age Range",
    int(data['age_mid'].min()),
    int(data['age_mid'].max()),
    (30, 80)
)

selected_specialty = st.sidebar.multiselect(
    "Select Medical Specialty",
    options=sorted(data['medical_specialty'].unique()),
    default=sorted(data['medical_specialty'].unique())
)

filtered_data = data[
    (data['age_mid'] >= age_filter[0]) &
    (data['age_mid'] <= age_filter[1]) &
    (data['medical_specialty'].isin(selected_specialty))
]

# --- Compute treatment score BEFORE using it ---
filtered_data = compute_treatment_score(filtered_data)
# --- Compute treatment score BEFORE using it ---
filtered_data = compute_treatment_score(filtered_data)

# --- Map numeric scores to descriptive labels ---
score_mapping = {0: "Poor", 1: "Moderate", 2: "Good"}
filtered_data['treatment_score_label'] = filtered_data['treatment_score'].map(score_mapping)

# ==============================================
# PAGE 1: MAIN DASHBOARD
# ==============================================
if page == "Main Dashboard":
    # ----------------------------------------------
    # Layout: Key Metrics including treatment score
    # ----------------------------------------------
    total_patients = len(filtered_data)
    readmitted_count = filtered_data['readmitted'].sum()
    avg_stay = filtered_data['time_in_hospital'].mean()
    avg_treatment_score = filtered_data['treatment_score'].mean(skipna=True)  # skip None


    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    col1.metric("Total Patients", total_patients)
    col2.metric("Readmissions", int(readmitted_count))
    col3.metric("Avg. Hospital Stay (days)", round(avg_stay, 2))
    col4.metric("Avg. Treatment Score", round(avg_treatment_score, 2))

    st.subheader("Treatment Effectiveness Insights")

    fig_score_dist = px.histogram(
        filtered_data,
        x='treatment_score_label',  # use descriptive labels
        title="Distribution of Treatment Scores",
        labels={'treatment_score_label':'Treatment Effectiveness'},
        color_discrete_sequence=custom_colors
    )
    st.plotly_chart(fig_score_dist, use_container_width=True)

    # ----------------------------------------------
    # Visualizations in Grid Layout (with full titles & labels)
    # ----------------------------------------------
    st.markdown("---")
    st.subheader("Exploratory Insights")

    # First row: 3 Plotly charts
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        # Age Distribution Chart
        fig_age = px.histogram(
            filtered_data,
            x='age_mid',
            nbins=10,
            color='readmitted',
            color_discrete_sequence=custom_colors,
            title="Age Distribution by Readmission",
            labels={'age_mid': 'Age (midpoint)', 'readmitted': 'Readmitted'}
        )
        st.plotly_chart(fig_age, width='stretch')

    with col2:
        # Readmission Rate by Medical Specialty
        fig_specialty = px.bar(
            filtered_data.groupby('medical_specialty')['readmitted'].mean().reset_index(),
            x='medical_specialty',
            y='readmitted',
            color='medical_specialty',
            color_discrete_sequence=custom_colors,
            title="Readmission Rate by Medical Specialty",
            labels={'readmitted': 'Readmission Rate', 'medical_specialty': 'Specialty'}
        )
        fig_specialty.update_layout(showlegend=False)
        st.plotly_chart(fig_specialty, width='stretch')

    with col3:
        # Hospital Stay vs Medications
        fig_stay = px.scatter(
            filtered_data,
            x='time_in_hospital',
            y='n_medications',
            color='readmitted',
            color_discrete_sequence=custom_colors,
            title="Relationship between Hospital Stay and Medications",
            labels={'time_in_hospital': 'Time in Hospital (days)', 'n_medications': 'Number of Medications'}
        )
        st.plotly_chart(fig_stay, width='stretch')

    # Second row: 2 Matplotlib/Seaborn charts
    top_specialties = filtered_data['medical_specialty'].value_counts().head(10)
    readmission_med = filtered_data.groupby('n_medications')['readmitted'].mean().reset_index()

    col4, col5 = st.columns(2, gap="medium")

    with col4:
        st.subheader("Top 10 Medical Specialties by Patient Count")
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        sns.barplot(x=top_specialties.values, y=top_specialties.index, palette='crest', ax=ax4)
        ax4.set_xlabel("Number of Patients")
        ax4.set_ylabel("Medical Specialty")
        ax4.set_title("Top 10 Medical Specialties by Patient Count")
        st.pyplot(fig4)

    with col5:
        st.subheader("Readmission Rate vs Number of Medications")
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        sns.lineplot(data=readmission_med, x='n_medications', y='readmitted', color='#00B2A9', ax=ax5)
        ax5.set_xlabel("Number of Medications")
        ax5.set_ylabel("Readmission Rate")
        ax5.set_title("Readmission Rate vs Number of Medications")
        st.pyplot(fig5)

    # Third row: 2 Plotly charts

    col6, col7 = st.columns(2, gap="medium")

    with col6:
        st.subheader("Diagnosis Distribution")

        # Melt diagnosis columns for scatter plot
        diag_cols = ['diag_1', 'diag_2', 'diag_3']
        diag_melt = filtered_data[diag_cols].reset_index().melt(id_vars='index', var_name='Diagnosis_Type', value_name='Diagnosis_Code')

        # Remove missing/NULL codes
        diag_melt = diag_melt.dropna(subset=['Diagnosis_Code'])

        fig_diag_scatter = px.scatter(
            diag_melt,
            x='Diagnosis_Type',
            y='Diagnosis_Code',
            color='Diagnosis_Type',
            title="Distribution of Diagnosis Codes",
            labels={
                'Diagnosis_Type': 'Diagnosis Category',
                'Diagnosis_Code': 'Diagnosis Code'
            },
            opacity=0.7,
            color_discrete_sequence=custom_colors
        )
        st.plotly_chart(fig_diag_scatter, use_container_width=True)

    with col7:
        st.subheader("Average Length of Stay by Diagnosis")

        # Get all diagnosis codes
        diag_long = filtered_data[['time_in_hospital'] + diag_cols].melt(
            id_vars='time_in_hospital',
            value_name='Diagnosis_Code'
        )

        diag_long = diag_long.dropna(subset=['Diagnosis_Code'])

        # Compute avg length of stay
        avg_stay_diag = diag_long.groupby('Diagnosis_Code')['time_in_hospital'].mean().reset_index()

        fig_avg_stay = px.bar(
            avg_stay_diag,
            x='Diagnosis_Code',
            y='time_in_hospital',
            color='Diagnosis_Code',
            title="Average Hospital Stay by Diagnosis Code",
            labels={
                'Diagnosis_Code': 'Diagnosis Code',
                'time_in_hospital': 'Average Length of Stay (days)'
            },
            color_discrete_sequence=custom_colors
        )

        fig_avg_stay.update_layout(showlegend=False)
        st.plotly_chart(fig_avg_stay, use_container_width=True)



# ==============================================
# PAGE 2: CORRELATION ANALYSIS
# ==============================================
elif page == "Correlation Analysis":
    st.subheader("📊 Correlation Analysis Dashboard")
    st.markdown("**Explore relationships between numerical variables in the dataset**")
    
    # Get numerical columns
    numerical_cols = filtered_data.select_dtypes(include=[np.number]).columns.tolist()
    
    # ----------------------------------------------
    # Correlation Heatmap
    # ----------------------------------------------
    st.markdown("---")
    st.subheader("Correlation Heatmap")
    
    # Calculate correlation matrix
    corr_matrix = filtered_data[numerical_cols].corr()
    
    # Create interactive heatmap
    fig_heatmap = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='RdBu_r',
        aspect="auto",
        title="Correlation Matrix of Numerical Features",
        zmin=-1,
        zmax=1
    )
    fig_heatmap.update_layout(height=600)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # ----------------------------------------------
    # Top Correlations with Readmission
    # ----------------------------------------------
    st.markdown("---")
    st.subheader("Top Correlations with Readmission")
    
    # Get correlations with readmitted
    readmit_corr = corr_matrix['readmitted'].drop('readmitted').sort_values(ascending=False)
    
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.markdown("**Positive Correlations**")
        positive_corr = readmit_corr[readmit_corr > 0].head(5)
        fig_pos = px.bar(
            x=positive_corr.values,
            y=positive_corr.index,
            orientation='h',
            title="Top 5 Positive Correlations with Readmission",
            labels={'x': 'Correlation Coefficient', 'y': 'Variable'},
            color=positive_corr.values,
            color_continuous_scale='Greens'
        )
        fig_pos.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_pos, use_container_width=True)
    
    with col2:
        st.markdown("**Negative Correlations**")
        negative_corr = readmit_corr[readmit_corr < 0].tail(5).sort_values()
        fig_neg = px.bar(
            x=negative_corr.values,
            y=negative_corr.index,
            orientation='h',
            title="Top 5 Negative Correlations with Readmission",
            labels={'x': 'Correlation Coefficient', 'y': 'Variable'},
            color=negative_corr.values,
            color_continuous_scale='Reds'
        )
        fig_neg.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_neg, use_container_width=True)
    
    # ----------------------------------------------
    # Pairwise Scatter Plots
    # ----------------------------------------------
    st.markdown("---")
    st.subheader("Pairwise Relationship Explorer")
    
    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("Select X Variable", numerical_cols, index=0)
    with col2:
        y_var = st.selectbox("Select Y Variable", numerical_cols, index=1)
    
    # Create scatter plot
    fig_scatter = px.scatter(
        filtered_data,
        x=x_var,
        y=y_var,
        color='readmitted',
        color_discrete_sequence=custom_colors,
        title=f"Relationship between {x_var} and {y_var}",
        labels={x_var: x_var.replace('_', ' ').title(), 
                y_var: y_var.replace('_', ' ').title()},
        opacity=0.6,
        trendline="ols"
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # ----------------------------------------------
    # Correlation Statistics Table
    # ----------------------------------------------
    st.markdown("---")
    st.subheader("Correlation Statistics")
    
    # Create a formatted correlation table
    corr_df = pd.DataFrame({
        'Variable 1': [],
        'Variable 2': [],
        'Correlation': [],
        'Strength': []
    })
    
    for i in range(len(numerical_cols)):
        for j in range(i+1, len(numerical_cols)):
            var1 = numerical_cols[i]
            var2 = numerical_cols[j]
            corr_val = corr_matrix.loc[var1, var2]
            
            # Determine strength
            if abs(corr_val) >= 0.7:
                strength = "Strong"
            elif abs(corr_val) >= 0.4:
                strength = "Moderate"
            else:
                strength = "Weak"
            
            corr_df = pd.concat([corr_df, pd.DataFrame({
                'Variable 1': [var1],
                'Variable 2': [var2],
                'Correlation': [round(corr_val, 3)],
                'Strength': [strength]
            })], ignore_index=True)
    
    # Sort by absolute correlation
    corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False).drop('Abs_Correlation', axis=1)
    
    st.dataframe(corr_df.head(20), use_container_width=True)

# ==============================================
# PAGE 3: VARIABLE ANALYSIS
# ==============================================
elif page == "Variable Analysis":
    st.subheader("📈 Variable Analysis Dashboard")
    st.markdown("**Deep dive into individual variable distributions and statistics**")
    
    # ----------------------------------------------
    # Variable Selection
    # ----------------------------------------------
    st.markdown("---")
    
    numerical_cols = filtered_data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = filtered_data.select_dtypes(include=['object']).columns.tolist()
    
    analysis_type = st.radio(
        "Select Analysis Type",
        ["Numerical Variables", "Categorical Variables"],
        horizontal=True
    )
    
    if analysis_type == "Numerical Variables":
        # ----------------------------------------------
        # Numerical Variable Analysis
        # ----------------------------------------------
        st.markdown("---")
        st.subheader("Numerical Variable Distribution")
        
        selected_var = st.selectbox("Select Variable to Analyze", numerical_cols)
        
        col1, col2, col3 = st.columns(3, gap="medium")
        
        with col1:
            # Distribution plot
            fig_dist = px.histogram(
                filtered_data,
                x=selected_var,
                nbins=30,
                title=f"Distribution of {selected_var.replace('_', ' ').title()}",
                labels={selected_var: selected_var.replace('_', ' ').title()},
                color_discrete_sequence=custom_colors,
                marginal="box"
            )
            fig_dist.update_layout(height=400)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Distribution by readmission status
            fig_dist_readmit = px.box(
                filtered_data,
                x='readmitted',
                y=selected_var,
                color='readmitted',
                title=f"{selected_var.replace('_', ' ').title()} by Readmission Status",
                labels={selected_var: selected_var.replace('_', ' ').title(),
                       'readmitted': 'Readmitted'},
                color_discrete_sequence=custom_colors
            )
            fig_dist_readmit.update_layout(height=400)
            st.plotly_chart(fig_dist_readmit, use_container_width=True)
        
        with col3:
            # Violin plot
            fig_violin = px.violin(
                filtered_data,
                y=selected_var,
                x='readmitted',
                color='readmitted',
                title=f"{selected_var.replace('_', ' ').title()} Distribution (Violin)",
                labels={selected_var: selected_var.replace('_', ' ').title()},
                color_discrete_sequence=custom_colors,
                box=True
            )
            fig_violin.update_layout(height=400)
            st.plotly_chart(fig_violin, use_container_width=True)
        
        # ----------------------------------------------
        # Statistical Summary
        # ----------------------------------------------
        st.markdown("---")
        st.subheader(f"Statistical Summary: {selected_var.replace('_', ' ').title()}")
        
        col1, col2, col3, col4, col5 = st.columns(5, gap="medium")
        
        stats = filtered_data[selected_var].describe()
        
        col1.metric("Mean", f"{stats['mean']:.2f}")
        col2.metric("Median", f"{stats['50%']:.2f}")
        col3.metric("Std Dev", f"{stats['std']:.2f}")
        col4.metric("Min", f"{stats['min']:.2f}")
        col5.metric("Max", f"{stats['max']:.2f}")
        
        # Detailed statistics table
        st.markdown("**Detailed Statistics**")
        stats_df = pd.DataFrame({
            'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median (50%)', '75%', 'Max'],
            'Value': [
                f"{stats['count']:.0f}",
                f"{stats['mean']:.2f}",
                f"{stats['std']:.2f}",
                f"{stats['min']:.2f}",
                f"{stats['25%']:.2f}",
                f"{stats['50%']:.2f}",
                f"{stats['75%']:.2f}",
                f"{stats['max']:.2f}"
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # ----------------------------------------------
        # Comparison Across Groups
        # ----------------------------------------------
        st.markdown("---")
        st.subheader("Comparison Across Medical Specialties")
        
        # Group by medical specialty
        specialty_stats = filtered_data.groupby('medical_specialty')[selected_var].agg(['mean', 'median', 'std']).reset_index()
        specialty_stats.columns = ['Medical Specialty', 'Mean', 'Median', 'Std Dev']
        specialty_stats = specialty_stats.sort_values('Mean', ascending=False)
        
        fig_specialty_comp = px.bar(
            specialty_stats,
            x='Medical Specialty',
            y='Mean',
            title=f"Average {selected_var.replace('_', ' ').title()} by Medical Specialty",
            labels={'Mean': f'Average {selected_var.replace("_", " ").title()}'},
            color='Mean',
            color_continuous_scale='Viridis'
        )
        fig_specialty_comp.update_layout(height=400)
        st.plotly_chart(fig_specialty_comp, use_container_width=True)
        
        st.dataframe(specialty_stats, use_container_width=True, hide_index=True)
    
    else:
        # ----------------------------------------------
        # Categorical Variable Analysis
        # ----------------------------------------------
        st.markdown("---")
        st.subheader("Categorical Variable Distribution")
        
        selected_var = st.selectbox("Select Variable to Analyze", categorical_cols)
        
        # Value counts
        value_counts = filtered_data[selected_var].value_counts().reset_index()
        value_counts.columns = [selected_var, 'Count']
        
        col1, col2 = st.columns(2, gap="medium")
        
        with col1:
            # Bar chart
            fig_bar = px.bar(
                value_counts,
                x=selected_var,
                y='Count',
                title=f"Distribution of {selected_var.replace('_', ' ').title()}",
                labels={selected_var: selected_var.replace('_', ' ').title()},
                color='Count',
                color_continuous_scale='Teal'
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Pie chart
            fig_pie = px.pie(
                value_counts,
                values='Count',
                names=selected_var,
                title=f"{selected_var.replace('_', ' ').title()} Proportion",
                color_discrete_sequence=px.colors.sequential.Teal
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # ----------------------------------------------
        # Frequency Table
        # ----------------------------------------------
        st.markdown("---")
        st.subheader("Frequency Table")
        
        value_counts['Percentage'] = (value_counts['Count'] / value_counts['Count'].sum() * 100).round(2)
        value_counts['Percentage'] = value_counts['Percentage'].astype(str) + '%'
        
        st.dataframe(value_counts, use_container_width=True, hide_index=True)
        
        # ----------------------------------------------
        # Relationship with Readmission
        # ----------------------------------------------
        st.markdown("---")
        st.subheader(f"{selected_var.replace('_', ' ').title()} vs Readmission")
        
        # Cross-tabulation
        crosstab = pd.crosstab(filtered_data[selected_var], filtered_data['readmitted'], normalize='index') * 100
        crosstab = crosstab.reset_index()
        crosstab.columns = [selected_var, 'Not Readmitted (%)', 'Readmitted (%)']
        
        # Stacked bar chart
        fig_stacked = px.bar(
            filtered_data,
            x=selected_var,
            color='readmitted',
            title=f"Readmission Rate by {selected_var.replace('_', ' ').title()}",
            labels={selected_var: selected_var.replace('_', ' ').title()},
            color_discrete_sequence=custom_colors,
            barmode='group'
        )
        fig_stacked.update_layout(height=400)
        st.plotly_chart(fig_stacked, use_container_width=True)
        
        st.dataframe(crosstab, use_container_width=True, hide_index=True)


# ----------------------------------------------
# Footer
# ----------------------------------------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; color: grey; font-size:14px;'>
        <img src='https://github.com/PersonUsername.png' width='30'> Created by Rufus Nwogu
    </p>
    """,
    unsafe_allow_html=True
)
st.caption("Developed using Streamlit and Plotly | Healthcare Dashboard")
