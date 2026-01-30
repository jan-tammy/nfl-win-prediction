#library imports
import streamlit as st
import pandas as pd
import joblib
import numpy as np

#App Setup
st.set_page_config(
    page_title="NFL Win Predictor",
    page_icon= "🏈",
    layout="centered"
)

st.title("NFL Home Win Predictor")
st.write("Select the **Home Team** and the **Away Team**. ")
st.write("The model uses recent team performance to predict the home team's win  probability.")

#Load artifacts
@st.cache_resource
def load_model():
    model = joblib.load("nfl_win_model.pkl")
    scaler = joblib.load("nfl_scaler.pkl")
    return model, scaler

@st.cache_data
def load_team_stats():
    return pd.read_csv("team_stats_latest.csv")

model, scaler = load_model()
team_stats = load_team_stats()

teams = sorted(team_stats["team"].unique())

#User inputs
home_team = st.selectbox("Home Team",teams)
away_team = st.selectbox("Away Team",teams)

if home_team == away_team:
    st.warning("Home and Away teams must be different.")
    st.stop()


#Feature Construction
def build_features(home_team, away_team):
  #Get latest stats
  home_stats = team_stats[team_stats["team"] == home_team].iloc[0]
  away_stats = team_stats[team_stats["team"] == away_team].iloc[0]

  #compute diff features
  features = pd.DataFrame({
      "diff_points_for_avg_last_5": [home_stats['points_for_avg_last_5'] - away_stats['points_for_avg_last_5']],
      "diff_points_against_avg_last_5": [home_stats['points_against_avg_last_5'] - away_stats['points_against_avg_last_5']],
      "diff_yardsPerPlay_avg_last_5": [home_stats['yardsPerPlay_avg_last_5'] - away_stats['yardsPerPlay_avg_last_5']],
      "diff_possessionTime_avg_last_5": [home_stats['possessionTime_avg_last_5'] - away_stats['possessionTime_avg_last_5']],
      "diff_firstDowns_avg_last_5": [home_stats['firstDowns_avg_last_5'] - away_stats['firstDowns_avg_last_5']],
      "diff_redZoneAttempts_avg_last_5": [home_stats['redZoneAttempts_avg_last_5'] - away_stats['redZoneAttempts_avg_last_5']],
      "diff_turnovers_avg_last_5": [home_stats['turnovers_avg_last_5'] - away_stats['turnovers_avg_last_5']]    
  })
  return features


#Prediction
if st.button("Predict Home Win Probability"):
    X = build_features(home_team,away_team)
    X_scaled = scaler.transform(X)

    prob = model.predict_proba(X_scaled)[:,1][0]

    st.markdown("## Prediction")
    st.metric(
        label = f"{home_team} Win Probability",
        value=f"{prob*100:0.1f}%"
    )

    #Confidence Interpretation
    if prob >= 0.65:
        st.success("Strong Home Team advantage.")
    elif prob>= 0.55:
        st.info("Slight Home Team advantage.")
    else:
        st.warning("Away team is competitive.")


    #Explainability
    st.markdown("## Feature Impact")

    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Value":X.iloc[0].values,
        "Coefficient":model.coef_[0]
    })

    coef_df["Impact"] = coef_df["Value"] * coef_df["Coefficient"]
    coef_df = coef_df.sort_values("Impact",key=abs, ascending=False)

    st.dataframe(
        coef_df[["Feature","Impact"]].style.format({"Impact":"{:.3f}"})
    )

    st.caption("Positive impact favors the home team, negative impact favors the away team.")