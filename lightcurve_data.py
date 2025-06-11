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
import os


df = pd.read_csv('toi-data-caltech-updated.csv')
new_pattern = r'\b(APC|FA|FP|PC)\b'

download_dir = "fresh_downloads"
os.makedirs(download_dir, exist_ok=True)

#for dframe in df['TIC ID']:
for i in range(0, 10):
    myTestTic = "TIC " + str(df['TIC ID'][i])
    print("TIC Id chosen is :" + myTestTic)
    lcs = search_lightcurve(myTestTic, mission="TESS", author="SPOC").download_all(download_dir=download_dir)
    # testdf = [lc.to_pandas() for lc in lcs]
    count = 0
    pandas_dfs = []
    pandas_df = []
    for lc in lcs:
        print('right after lc in lcs')
        pandas_df = []
        if lc is not None:
            pandas_df = lc.to_pandas() #Converting to pandas data frame
            print('Converted to pandas_df')
            if lc is not None:  # Check if lc is a valid LightCurve object
                if 'time' not in pandas_df.columns:
                    pandas_df = pandas_df.reset_index() #Go back to beginning
        pandas_dfs.append(pandas_df)
        print('Appended pandas_df to pandas_dfs')
        #combined_df = pd.concat(pandas_dfs, ignore_index=True)
        print('Printing all columns from lightcurve data')
        #print(combined_df.head())
        print(pandas_df.head(5))
        
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

        print("TIC Id chosen is :" + myTestTic)
        #plt.figure(figsize=(12, 5))
        plt.plot(pandas_df['time'], pandas_df['flux'], label='Flux', alpha=0.5)
        plt.scatter(anomaly_times, anomaly_fluxes, color='red', label='Anomalies', s=10)
        plt.title("Isolation Forest Anomalies")
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

        # bls = lc_clean.to_periodogram(method="bls", period=np.linspace(0.5, 20, 10000))
        # bls.plot()
        # plt.show()
        # lombscargle = lc_clean.to_periodogram(method="lombscargle", period=np.linspace(0.5, 20, 10000))
        # best_period = lombscargle.period_at_max_power
        # spike_time = lc_clean.time[np.argmax(lc_clean.flux)]
        # folded = lc_clean.fold(period=best_period, epoch_time=spike_time)
        # folded.plot()

        lscargle = lc_clean.to_periodogram(method="lombscargle", period=np.linspace(0.5, 20, 10000))
        lscargle.plot()
        plt.title(f"Lomb-Scargle Periodogram")
        plt.grid(True)
        plt.show()

        best_period = lscargle.period_at_max_power
        spike_time = lc_clean.time[np.argmax(lc_clean.flux)]
        folded = lc_clean.fold(period=best_period, epoch_time=spike_time)
        folded.plot()
        plt.title(f"Folded at P = {best_period:.5f} days, epoch = spike_time")
        plt.grid(True)
        filename = str(myTestTic) + "_" + str(count) + '_folded.png'
        count += 1
        plot_filename = os.path.join('/Users/avanijavangala/Downloads/CCIR_Project/Anomaly_Detection_Exoplanet', filename)
        plt.savefig(plot_filename)
        plt.show()

        # df.loc[df['TFOPWG Disposition'].astype(str).str.contains(new_pattern, case=False, na=False, regex=True)].index
        # spike_time = lc_clean.time[np.argmax(lc_clean.flux)]
        # folded = lc_clean.fold(period=best_period, epoch_time=spike_time)
        # folded.plot()
        # plt.title(f"Folded at P = {best_period:.5f} days, epoch = spike_time")
        # plt.grid(True)
        # plt.show()


        # Fold the light curve using the best period
        #folded = lc_clean.fold(period=2 * bls.period_at_max_power)

        # Plot it
        #folded.plot()
        #plt.show()