import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from datetime import datetime
import streamlit as st
import time
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import datetime as dt
import plotly.express as px
from functions.funcs import r_score, fm_score, segmenting_customers


#Title & Description Of Adoption Report
st.title("Performing Prediction and Analysis")

#CSS Markdown for border of graph
def _max_width_():
    st.markdown(
        f"""
        <style>
        [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"]  {{
            display: grid;
            border: 1px solid white;
            padding: 20px;
            border-radius: 10px;
            color: #003366;
            margin-bottom: 5px;
            min-height: 300px;
            align-items: center;
        }}
        <style>
    """,
        unsafe_allow_html=True,
    )

#Specifies Spinner, for spinning untill results fetched
with st.spinner("Fetching Data..."):

    df = pd.DataFrame()

    #SideBar Form 1
    with st.sidebar:
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

    #Setting width
    _max_width_()

    if len(df) != 0:

        st.write("##### Sample Dataset")
        st.dataframe(df.head())

        df['transaction_time'] = pd.to_datetime(df['transaction_time'])
        snapshot_date = dt.datetime(2021,10,30,23,20,20)

        data = df.groupby(['customer_id'], as_index=False).agg(
            recency = ('transaction_time', lambda x: (snapshot_date - x.max()).days),
            frequency = ('transaction_time', lambda x: len(x)),
            monetary_value = ('value', lambda x: x.sum())
        )
        st.write("##### Calculated RFM values for each customer.")
        st.dataframe(data.head())

        quantiles = data.quantile(q=[0.25,0.5,0.75])
        quantiles = quantiles.to_dict()

        segmented_rfm = data

        segmented_rfm['r_quartile'] = segmented_rfm['recency'].apply(r_score, args=('recency',quantiles,))
        segmented_rfm['f_quartile'] = segmented_rfm['frequency'].apply(fm_score, args=('frequency',quantiles,))
        segmented_rfm['m_quartile'] = segmented_rfm['monetary_value'].apply(fm_score, args=('monetary_value',quantiles,))
        st.write("##### Calculated Quartiles values for each customer.")
        st.dataframe(segmented_rfm.head())

        segmented_rfm['rfm_segment'] = segmented_rfm.r_quartile.map(str) + segmented_rfm.f_quartile.map(str) + segmented_rfm.m_quartile.map(str)
        segmented_rfm['rfm_score'] = segmented_rfm.r_quartile + segmented_rfm.f_quartile + segmented_rfm.m_quartile
        st.write("##### Final dataframe with RFM score assigned to each customer.")
        st.dataframe(segmented_rfm.head())

        scores_ploting = segmented_rfm.groupby(['rfm_score'], as_index=False)[['recency', 'frequency', 'monetary_value']].mean()

        st.write("##### Trend of recency, frequency, and mortage values over the RFM scores.")
        fig = plt.figure(figsize=(12,4))
        plt.plot(scores_ploting['rfm_score'], scores_ploting['recency'], label='recency')    
        plt.plot(scores_ploting['rfm_score'], scores_ploting['frequency'], label='frequency')    
        plt.plot(scores_ploting['rfm_score'], scores_ploting['monetary_value'], label='monetary_value') 
        plt.xlabel('RFM scores')
        plt.ylabel('Trend')
        plt.legend()   
        st.pyplot(fig)

        segmented_rfm['category'] = segmented_rfm['rfm_score'].apply(segmenting_customers)
        segment_rfm = segmented_rfm.groupby(['category'], as_index=False)[['recency', 'frequency', 'monetary_value']].mean()

        st.write("##### Identify the types of users over the RFM Categories.")
        fig = plt.figure(figsize=(12,4))     
        plt.bar(segment_rfm['category'], segment_rfm['monetary_value'], width = 0.25, edgecolor = 'black', label='monetary_value')  
        plt.bar(segment_rfm['category'], segment_rfm['recency'], width = 0.25, edgecolor = 'black', label='recency')
        plt.bar(segment_rfm['category'], segment_rfm['frequency'], width = 0.25, edgecolor = 'black', label='frequency') 
        plt.xlabel("RFM Categories as Top(9-12), Middle(5-8), Low(<5)")
        plt.ylabel("Trend")
        plt.title("")  
        plt.legend()
        plt.show()
        st.pyplot(fig)

        #Kmeans

        ss = StandardScaler()

        segmented_rfm['recency_log'] = segmented_rfm['recency'].apply(lambda x: np.log(x))
        segmented_rfm['frequency_log'] = segmented_rfm['frequency'].apply(lambda x: np.log(x))
        segmented_rfm['monetary_value_log'] = segmented_rfm['monetary_value'].apply(lambda x: np.log(x))

        X = segmented_rfm[['recency_log', 'frequency_log', 'monetary_value_log']]
        X_norm = ss.fit_transform(X)

        normdf = pd.DataFrame(data=X_norm, columns=['r', 'f', 'm'])

        st.write("##### Elbow method for selecting no. of clusters, which help in reducing loss drastically.")
        sse = {}
        for k in range(1, 4):
            kmeans = KMeans(n_clusters=k, n_init=10)
            kmeans.fit(X_norm)
            sse[k] = kmeans.inertia_

        fig = plt.figure(figsize=(12,4))
        plt.xlabel('k'); 
        plt.ylabel('Sum of squared errors')
        sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
        plt.show()
        st.pyplot(fig)

        segmented_rfm['clusters'] = kmeans.labels_
        segmented_rfm.groupby(['clusters'])[['recency', 'frequency', 'monetary_value']].mean()
        kmeans_segments = segmented_rfm.groupby(['clusters'], as_index=False)[['recency', 'frequency', 'monetary_value']].mean()

        st.write("##### Identify the types of users over the k-means clusters.")
        fig = plt.figure(figsize=(12,4))     
        plt.bar(kmeans_segments['clusters'], kmeans_segments['monetary_value'], width = 0.25, edgecolor = 'black', label='monetary_value')  
        plt.bar(kmeans_segments['clusters'], kmeans_segments['recency'], width = 0.25, edgecolor = 'black', label='recency')
        plt.bar(kmeans_segments['clusters'], kmeans_segments['frequency'], width = 0.25, edgecolor = 'black', label='frequency') 
        plt.xlabel("Clusters")
        plt.ylabel("Trend")
        plt.legend()
        plt.show()
        st.pyplot(fig)

        st.write("##### Clusters ploted in 3d chart based on recency, frequency, and mortage values.")
        fig = px.scatter_3d (
            segmented_rfm, 
            x='recency', 
            y='frequency',
            z='monetary_value',
            color='clusters', opacity = 0.8, size_max=30
        )
        fig.show()

        st.plotly_chart(fig, use_container_width=True)

        #predictor
        tagging = df
        snapshot = dt.datetime(2021,10,30)

        tagging = tagging.groupby(['customer_id'], as_index=False).agg(
            acquired_date = ('acquired_date', lambda x: x.min())
        )

        tagging['acquired_date'] = tagging['acquired_date'].apply(lambda x: x.split(' ')[0])
        tagging['acquired_date'] = pd.to_datetime(tagging['acquired_date'])
        tagging['recency'] = (snapshot - tagging['acquired_date'])
        tagging['recency'] = tagging['recency'].astype(str)
        tagging['recency'] = tagging['recency'].apply(lambda x: x.split(' ')[0])
        tagging['recency'] = tagging['recency'].astype(int)

        tagging['acquired_month'] = tagging['acquired_date'].dt.month
        tagging['acquired_year'] = tagging['acquired_date'].dt.year

        tagging = tagging.groupby(['acquired_year', 'acquired_month'], as_index=False).agg(
            users_in_each_month = ('customer_id', lambda x: x.count())
        )

        recent_2021 = tagging[tagging['acquired_year'] == 2021][['acquired_month', 'users_in_each_month']]

        st.write("##### Checking out the no. acquired users of 2021 by segmenting months, It'll help us to check no. of newly acquired users by looking into recent month.")
        fig = plt.figure(figsize=(12,4))     
        plt.bar(recent_2021['acquired_month'], recent_2021['users_in_each_month'], width = 0.25, edgecolor = 'black', label='users in each month')  
        plt.xlabel("Months(2021)")
        plt.ylabel("No. of acquired users.")
        plt.legend()
        plt.show()
        st.pyplot(fig)
