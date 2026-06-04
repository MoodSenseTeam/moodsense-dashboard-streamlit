import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# config
st.set_page_config(
    page_title="MoodSense Dashboard",
    page_icon="",
    layout="wide"
)

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#2c3e50'

# load data
df = pd.read_csv('mental_health_final.csv')

# header
st.title("MoodSense Dashboard")
st.markdown(
    "Dashboard visualisasi kesehatan mental mahasiswa berbasis *data analysis* dan *mood prediction*."
)
st.markdown("---")

# sidebar
st.sidebar.header("Filter Data")

selected_gender = st.sidebar.multiselect(
    "Pilih Gender",
    options=df['Gender'].unique(),
    default=df['Gender'].unique()
)

filtered_df = df[df['Gender'].isin(selected_gender)]


st.subheader("Statistik Utama")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Responden", f"{len(filtered_df):,}")

with col2:
    st.metric(
        "Avg Stress",
        f"{filtered_df['Stress_Level'].mean():.2f}"
    )

with col3:
    st.metric(
        "Avg Depression",
        f"{filtered_df['Depression_Score'].mean():.2f}"
    )

with col4:
    st.metric(
        "Avg Anxiety",
        f"{filtered_df['Anxiety_Score'].mean():.2f}"
    )

st.markdown("---")


st.subheader("Distribusi Skor Kesehatan Mental")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))


sns.histplot(filtered_df['Stress_Level'], kde=True, ax=axes[0], color="#5c7cfa")
axes[0].set_title('Stress Level Distribution', fontsize=12, pad=10)
axes[0].set_xlabel('Score')

sns.histplot(filtered_df['Depression_Score'], kde=True, ax=axes[1], color="#ae3ec9")
axes[1].set_title('Depression Score Distribution', fontsize=12, pad=10)
axes[1].set_xlabel('Score')

sns.histplot(filtered_df['Anxiety_Score'], kde=True, ax=axes[2], color="#0ca678")
axes[2].set_title('Anxiety Score Distribution', fontsize=12, pad=10)
axes[2].set_xlabel('Score')

plt.tight_layout()
st.pyplot(fig)


# korelasi
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Korelasi Antar Skor")
    fig, ax = plt.subplots(figsize=(7, 5))
    
    sns.heatmap(
        filtered_df[['Stress_Level', 'Depression_Score', 'Anxiety_Score']].corr(),
        annot=True,
        cmap='vlag', 
        vmin=-1, vmax=1,
        center=0,
        linewidths=.5,
        ax=ax
    )
    ax.set_title('Correlation Matrix', fontsize=12, pad=10)
    st.pyplot(fig)

with col_right:
    st.subheader("Mental Health Index")
    fig, ax = plt.subplots(figsize=(7, 5))
    
    sns.histplot(
        filtered_df['Mental_Health_Index'],
        kde=True,
        color='#1098ad',
        ax=ax
    )
    ax.set_title('Mental Health Index Distribution', fontsize=12, pad=10)
    st.pyplot(fig)


# eksternal
st.subheader("Analisis Faktor Eksternal")
col_fact1, col_fact2 = st.columns(2)

with col_fact1:
    st.markdown("#### Kualitas Tidur vs Tingkat Stress")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    sns.boxplot(
        x='Sleep_Quality',
        y='Stress_Level',
        data=filtered_df,
        palette='crest',
        ax=ax
    )
    ax.set_title('Sleep Quality vs Stress Level', fontsize=11)
    st.pyplot(fig)

with col_fact2:
    st.markdown("#### Tekanan Finansial vs Depresi")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    sns.boxplot(
        x='Financial_Stress',
        y='Depression_Score',
        data=filtered_df,
        palette='flare',
        ax=ax
    )
    ax.set_title('Financial Stress vs Depression Score', fontsize=11)
    st.pyplot(fig)


# label mood
st.subheader("Distribusi Kategori Mood")

fig, ax = plt.subplots(figsize=(10, 4.5))
mood_counts = filtered_df['Mood_Label'].value_counts()

sns.countplot(
    x='Mood_Label',
    data=filtered_df,
    order=mood_counts.index,
    palette='Set2',
    ax=ax
)

ax.set_title('Mood Category Distribution', fontsize=12, pad=10)
ax.set_xlabel('Mood Label')
ax.set_ylabel('Jumlah Responden')


max_val = mood_counts.max()
for i, v in enumerate(mood_counts.values):
    ax.text(i, v + (max_val * 0.01), str(v), ha='center', fontsize=10, fontweight='bold', color='#2c3e50')

sns.despine(ax=ax)
st.pyplot(fig)

# view data
st.subheader("Dataset Preview")
st.dataframe(filtered_df.head(20), use_container_width=True)

# footer
st.markdown('---')
st.caption('MoodSense © 2026 | Mental Health Monitoring Dashboard')