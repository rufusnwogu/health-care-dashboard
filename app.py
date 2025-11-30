
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
    data_path = "Dataset/cleaned_hospital_data.csv"  
    df = pd.read_csv(data_path)
    return df

data = load_data()

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

# ----------------------------------------------
# Layout: 3 Key Metrics
# ----------------------------------------------
total_patients = len(filtered_data)
readmitted_count = filtered_data['readmitted'].sum()
avg_stay = filtered_data['time_in_hospital'].mean()

# Combine all diagnosis columns to find the most common overall
diagnosis_cols = ['diag_1', 'diag_2', 'diag_3']
diagnosis_values = pd.concat([filtered_data[col] for col in diagnosis_cols if col in filtered_data], axis=0)
diagnosis_values = diagnosis_values.dropna()
if not diagnosis_values.empty:
    common_diag = diagnosis_values.mode()[0]
else:
    common_diag = "No Diagnosis Data"

# KPI Layout with Shadow Styling
col1, col2, col3, col4 = st.columns(4, gap= "medium")
col1.metric("Total Patients", total_patients)
col2.metric("Readmissions", int(readmitted_count))
col3.metric("Avg. Hospital Stay (days)", round(avg_stay, 2))
col4.metric("Most Common Diagnosis", common_diag)

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

# ----------------------------------------------
# Footer
# ----------------------------------------------
st.markdown("---")
st.caption("Developed using Streamlit and Plotly | Healthcare Dashboard")
