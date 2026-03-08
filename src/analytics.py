import pandas as pd
import streamlit as st

def show_fleet_graphs(drivers):

    risk_counts = drivers["risk"].value_counts()

    st.subheader("Fleet Risk Distribution")

    st.bar_chart(risk_counts)