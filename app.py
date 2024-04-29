import sys
import os
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

alt.themes.enable("dark")

##################################
# Page configuration
st.set_page_config(
    page_title="Tellco User Analytics",
    page_icon=":newspaper:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# path where the data is stored
data_path = 'features/'

#######################
# CSS styling
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)

#######################
# Load data
# df_reshaped = pd.read_csv('../data/findings/tags_count.csv')

#######################
# Sidebar
with st.sidebar:
    st.title(':newspaper: News analysis Dashboard')


    color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo',
                        'viridis']
    selected_color_theme = st.selectbox('Select a color theme', color_theme_list)


#######################

def load_data(file_path):
    news_data = pd.read_csv(file_path)
    return news_data


alt.themes.enable("dark")


def plot_mse():
    # Assuming you have a model, features (X) and target (y)
    model = RandomForestRegressor()
    X = np.random.rand(100, 4)
    y = np.random.rand(100)

    # Calculate the cross validation scores
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

    # Negate the scores as it is a convention in sklearn for error/loss metrics
    mse_scores = -scores

    # Create a DataFrame for the scores
    df_scores = pd.DataFrame(mse_scores, columns=['MSE'])

    mean_score = df_scores.mean()
    st.write(f'Mean MSE: {mean_score[0]}')

    # Plot the MSE scores
    st.line_chart(df_scores)


def plot_avg_satisfaction_score_per_cluster():
    # Load the data
    df_experience = pd.read_csv(data_path + 'user_satisfaction_scores.csv')

    # Calculate the satisfaction score as the average of the engagement score and the experience score
    df_experience['satisfaction_score'] = (df_experience['engagement_score'] + df_experience['experience_score']) / 2

    # Aggregate the average satisfaction and experience score per cluster
    average_scores = df_experience.groupby('cluster')[['satisfaction_score', 'experience_score']].mean()

    # Display the DataFrame in Streamlit
    st.dataframe(average_scores)

    # Set the style of matplotlib to dark background
    plt.style.use('dark_background')

    # Plot the average scores using matplotlib
    fig, ax = plt.subplots()
    average_scores.plot(kind='bar', ax=ax)
    plt.title('Average Satisfaction and Experience Score per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Score')

    # Display the plot in Streamlit
    st.pyplot(fig)


def main():
    st.title('Tellco User Analytics Dashboard')
    plot_mse()
    plot_avg_satisfaction_score_per_cluster()

if __name__ == '__main__':
    main()
