from lightkurve import search_lightcurve
from lightkurve import LightCurve
from lightkurve.periodogram import BoxLeastSquaresPeriodogram
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import seaborn as sns
from pyod.models.auto_encoder import AutoEncoder
from pyod.utils.data import generate_data
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from scipy.signal import find_peaks
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

for i in range (0, 10): 

    # df = pd.read_csv('toi-data-caltech.csv')

    # myTestTic = "TIC " + str(df['TIC ID'][1])
    # print("TIC Id chosen is :" + myTestTic)

    lcs = search_lightcurve("TIC 466376085", mission="TESS").download_all()

    pandas_dfs = []
    pandas_df = []
    for lc in lcs:
        if lc is not None:
            pandas_df = lc.to_pandas() #Converting to pandas data frame
            if lc is not None:  # Check if lc is a valid LightCurve object
                if 'time' not in pandas_df.columns:
                    pandas_df = pandas_df.reset_index() #Go back to beginning
        pandas_dfs.append(pandas_df)
        #print(pandas_df.head(5))
        window_size = 150
        X = []

        flux = pandas_df['flux'].values

        for i in range(len(flux) - window_size):
            window = flux[i:i + window_size]
            X.append(window)

        X = np.array(X)# Shape: (n_windows, window_size)


        model = IsolationForest(contamination=0.05, random_state=42)
        labels = model.fit_predict(X)
        scores = model.decision_function(X) 

        anomaly_times = pandas_df['time'].values[:len(labels)][labels == -1]
        anomaly_fluxes = pandas_df['flux'].values[:len(labels)][labels == -1]

        # indices = np.where(anomaly_times > 3605 and anomaly_times < 3610)

        plt.figure(figsize=(12, 5))
        plt.plot(pandas_df['time'], pandas_df['flux'], label='Flux', alpha=0.5)
        plt.scatter(anomaly_times, anomaly_fluxes, color='red', label='Anomalies', s=10)
        plt.title("Isolation Forest Anomalies (TIC 418262301)")
        plt.xlabel("Time (BJD - 2457000)")
        plt.ylabel("Normalized Flux")
        plt.legend()
        plt.grid(True)
        plt.show()

        pandas_df["flux_smooth"] = pandas_df["flux"].rolling(window=5, center=True).mean()
        pandas_df["flux_smooth"] = pandas_df["flux_smooth"].bfill()

        peaks, _ = find_peaks(-pandas_df["flux_smooth"].fillna(method="bfill"), prominence=0.005, distance=20)

        # Step 4: Extract windows around each dip
        window = 50
        segments = []
        peak_times = [] 
        for peak in peaks[:5]:
            start, end = peak - window, peak + window
            if start >= 0 and end < len(pandas_df):
                segment_flux = pandas_df["flux"].iloc[start:end].values
                segment_time = pandas_df["time"].iloc[start:end].values
                segments.append(segment_flux)
                peak_times.append(pandas_df['time'].iloc[peak])
                # plt.plot(segment_time, segment_flux, alpha=0.6)
        # plt.title("Flux Segments Around Detected Dips")
        # plt.xlabel("Time (days)")
        # plt.ylabel("Normalized Flux")
        # plt.grid(True)
        #plt.show()


        segments = np.array(segments)
        segments = segments[:, :, np.newaxis]

        scaler = TimeSeriesScalerMeanVariance()
        segments_scaled = scaler.fit_transform(segments)

        n_clusters = 3

        # Cluster dips using KShape
        ks = KShape(n_clusters=3, random_state=42)
        labels = ks.fit_predict(segments_scaled)

        x_axis = np.arange(-window, window)
        # fig, axs = plt.subplots(n_clusters, 1, figsize=(10, 3 * n_clusters))

        # for cluster_id in range(n_clusters):
        #     axs[cluster_id].set_title(f"KShape Cluster {cluster_id}")
        #     for i, label in enumerate(labels):
        #         if label == cluster_id:
        #             axs[cluster_id].plot(x_axis, segments_scaled[i].ravel(), alpha=0.5)
        #     axs[cluster_id].set_xlabel("Time (centered around dip)")
        #     axs[cluster_id].set_ylabel("Normalized Flux")
        #     axs[cluster_id].grid(True)

        #plt.tight_layout()
        #plt.show()

        # for i, center in enumerate(ks.cluster_centers_):
        #     plt.plot(center.ravel(), label=f"Cluster {i}")
        # plt.legend()
        # plt.title("Average Shape of Each Cluster")
        #plt.show()

        transit_cluster = 0

        real_transits = [peak_times[i] for i in range(len(labels)) if labels[i] == transit_cluster]
        potential_binaries = [peak_times[i] for i in range(len(labels)) if labels[i] != transit_cluster]

        # for i, center in enumerate(ks.cluster_centers_):
        #     plt.plot(center.ravel(), label=f"Cluster {i}")
        #plt.show()

        noisy_transit_times = [peak_times[i] for i in range(len(labels)) if labels[i] != transit_cluster]
        mask = np.ones(len(lc), dtype=bool)

        for t in noisy_transit_times:  # e.g., from peak_times of noisy clusters
            mask &= np.abs(lc.time.value - t) > 0.2  # remove ±0.2 days around each bad dip

        lc_clean = lc[mask]

        bls = lc_clean.to_periodogram(method="bls", period=np.linspace(0.5, 20, 10000))
        bls.plot()
        plt.show()

        best_period = bls.period_at_max_power

        # Fold the light curve using the best period
        folded = lc_clean.fold(period=2 * bls.period_at_max_power)

        # Plot it
        folded.plot()
        plt.show()