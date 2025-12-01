
# ==============================================
# Streamlit Dashboard for Hospital Readmission
# ==============================================

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio

# ----------------------------------------------
# Load Data
# ----------------------------------------------
@st.cache_data
def load_data():
    data_path = r"C:\Users\DELL\Documents\Health Care Dashboarb\Dataset\cleaned_hospital_data.csv"
    df = pd.read_csv(data_path)
    return df
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
