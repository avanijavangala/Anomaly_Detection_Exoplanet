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


#lcs = search_lightcurve("TIC 418262301", mission="TESS").download_all()

df = pd.read_csv('toi-data-caltech.csv')
print('Printing all columns from TOI data')
print(df['TIC ID'])
print(df.columns)
print(df.shape)
print(df.describe())
print(df['Comments'][1])

myTestTic = "TIC " + str(466376085)
print("TIC Id chosen is :" + myTestTic)

mask = (df['TIC ID']== myTestTic)

filtered_df = df[mask]
print('all the ones with same TIC ID')
print(filtered_df)

#testlc = search_lightcurve("TIC 418262301", mission="TESS").download()
testlc = search_lightcurve(myTestTic, mission="TESS").download()
testdf = testlc.to_pandas()
print('Printing all columns from lightcurve data')
print(testdf.head(10))
print(testdf.shape)
print(testdf.columns)
print(testdf.describe())

if 'time' not in testdf.columns:
    testdf = testdf.reset_index()

window_size = 500
X = []

flux = testdf['flux'].values

for i in range(len(flux) - window_size):
    window = flux[i:i + window_size]
    X.append(window)

X = np.array(X)  # Shape: (n_windows, window_size)

from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.05, random_state=42)
labels = model.fit_predict(X)
scores = model.decision_function(X) 

anomaly_times = testdf['time'].values[:len(labels)][labels == -1]
anomaly_fluxes = testdf['flux'].values[:len(labels)][labels == -1]

plt.figure(figsize=(12, 5))
plt.plot(testdf['time'], testdf['flux'], label='Flux', alpha=0.5)
plt.scatter(anomaly_times, anomaly_fluxes, color='red', label='Anomalies', s=10)
plt.title("Isolation Forest Anomalies (TIC 418262301)")
plt.xlabel("Time (BJD - 2457000)")
plt.ylabel("Normalized Flux")
plt.legend()
plt.grid(True)
plt.show()

print('From here we are doing some basic time series classification of anomalies')

testdf["flux_smooth"] = testdf["flux"].rolling(window=5, center=True).mean()
testdf["flux_smooth"] = testdf["flux_smooth"].bfill()

peaks, _ = find_peaks(-testdf["flux_smooth"].fillna(method="bfill"), prominence=0.005, distance=20)

# Step 4: Extract windows around each dip
window = 50
segments = []
peak_times = [] 
for peak in peaks[:5]:
    start, end = peak - window, peak + window
    if start >= 0 and end < len(df):
        segment_flux = testdf["flux"].iloc[start:end].values
        segment_time = testdf["time"].iloc[start:end].values
        segments.append(segment_flux)
        peak_times.append(testdf['time'].iloc[peak])
        plt.plot(segment_time, segment_flux, alpha=0.6)
plt.title("Flux Segments Around Detected Dips")
plt.xlabel("Time (days)")
plt.ylabel("Normalized Flux")
plt.grid(True)
plt.show()


segments = np.array(segments)
segments = segments[:, :, np.newaxis]

scaler = TimeSeriesScalerMeanVariance()
segments_scaled = scaler.fit_transform(segments)

n_clusters = 3

# Cluster dips using KShape
ks = KShape(n_clusters=3, random_state=42)
labels = ks.fit_predict(segments_scaled)

x_axis = np.arange(-window, window)
fig, axs = plt.subplots(n_clusters, 1, figsize=(10, 3 * n_clusters))

for cluster_id in range(n_clusters):
    axs[cluster_id].set_title(f"KShape Cluster {cluster_id}")
    for i, label in enumerate(labels):
        if label == cluster_id:
            axs[cluster_id].plot(x_axis, segments_scaled[i].ravel(), alpha=0.5)
    axs[cluster_id].set_xlabel("Time (centered around dip)")
    axs[cluster_id].set_ylabel("Normalized Flux")
    axs[cluster_id].grid(True)

plt.tight_layout()
plt.show()

for i, center in enumerate(ks.cluster_centers_):
    plt.plot(center.ravel(), label=f"Cluster {i}")
plt.legend()
plt.title("Average Shape of Each Cluster")
plt.show()

transit_cluster = 0

real_transits = [peak_times[i] for i in range(len(labels)) if labels[i] == transit_cluster]
potential_binaries = [peak_times[i] for i in range(len(labels)) if labels[i] != transit_cluster]

for i, center in enumerate(ks.cluster_centers_):
    plt.plot(center.ravel(), label=f"Cluster {i}")
plt.show()

noisy_transit_times = [peak_times[i] for i in range(len(labels)) if labels[i] != transit_cluster]
mask = np.ones(len(testlc), dtype=bool)

for t in noisy_transit_times:  # e.g., from peak_times of noisy clusters
    mask &= np.abs(testlc.time.value - t) > 0.2  # remove ±0.2 days around each bad dip

lc_clean = testlc[mask]

bls = lc_clean.to_periodogram(method="bls", period=np.linspace(0.5, 20, 10000))
bls.plot()
plt.show()

best_period = bls.period_at_max_power

# Fold the light curve using the best period
folded = lc_clean.fold(period=2 * bls.period_at_max_power)

# Plot it
plavchan = lc_clean.to_periodogram(method="lombscargle", period=np.linspace(0.5, 20, 10000))
plavchan.plot()
plt.title(f"Plavchan Periodogram")
plt.grid(True)
plt.show()

best_period = plavchan.period_at_max_power

# Fold the light curve using the best period
fig,ax=plt.subplots(figsize=(10,6))
folded1 = lc_clean.fold(period=plavchan.period_at_max_power)
folded2 = lc_clean.fold(period=2*plavchan.period_at_max_power)
folded3 = lc_clean.fold(period=0.5*plavchan.period_at_max_power)
# Plot it
folded1.plot(ax=ax,label="Folded 1P",color='blue', linestyle='-')
folded2.plot(ax=ax,label="Folded 2P", color='red', linestyle='--')
folded3.plot(ax=ax,label="Folded 0.5P",color='green', linestyle=':')
plt.title(f"Folded Light Curve at {plavchan.period_at_max_power:.4f} days")
plt.grid(True)
plt.show()
# print('Columns in lightcurve data::')
# print(testdf.columns)

# ##Plot everything against everything 
# g = sns.PairGrid(testdf)
# g.map(sns.scatterplot)
# g.add_legend()
# plt.show()

# # now getting correlation between all parameters in the lightcurves
# numeric_df = testdf.select_dtypes(include=np.number)
# corr = numeric_df.corr()
# print(corr)

# mask = np.array(corr)
# mask[np.tril_indices_from(mask)] = False
# sns.heatmap(corr, vmax=.5, mask=mask, square=True, cbar=True)
# plt.xticks(rotation=45)
# plt.show()

# # ## Define the threshold
# # #threshold_upper = 0.2
# # #threshold_lower = -0.4
# # #threshold = 0.2

# # #mask = ((corr >= threshold_upper) | (corr <= threshold_lower)) 
# # ##mask = abs(corr) > threshold
# # #keep = mask.any(axis=1)

# # ## Filter the correlation matrix
# # #filtered_corr = corr.loc[keep, keep]

# # newmask = (np.triu(np.ones_like(corr, dtype=bool)))

# # #Visualize the correlation
# # plt.figure(figsize=(28, 24))
# # mask = np.array(corr)
# # mask[np.tril_indices_from(mask)] = False
# # sns.heatmap(corr, mask=newmask, vmax=0.5, square=True, annot=True, cmap='seismic', fmt=".2f", linewidths=0.5)
# # plt.title('Correlation of TESS lightcurve data')
# # plt.xticks(rotation=90)
# # plt.show()

# if 'time' not in testdf.columns:
#     testdf = testdf.reset_index()
# lctest = LightCurve(time=testdf['time'], flux=testdf['flux'])
# timevals = lctest.time.value
# fluxvals = lctest.flux.value
# flat_lc = lctest.flatten()
# flat_flux_vals = flat_lc.flux.value

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
# ax1.plot(timevals, fluxvals, color='blue', alpha=0.6)
# ax1.set_title('Original Flux')
# ax1.set_ylabel('Flux')
# ax1.grid(True)
# ax2.plot(timevals, flat_flux_vals, color='green', alpha=0.6)
# ax2.set_title('Flattened Flux')
# ax2.set_xlabel('Time (BJD - 2457000)')
# ax2.set_ylabel('Flux')
# ax2.grid(True)
# plt.tight_layout()
# plt.show()

# test_flux = testdf['flux'].values
# test_time = testdf['time'].values

# feature_list = []

# features = {
#     'mean_flux': np.mean(test_flux),
#         'std_flux': np.std(test_flux),
#         'min_flux': np.min(test_flux),
#         'max_flux': np.max(test_flux),
#         'flux_range': np.max(test_flux) - np.min(test_flux),
#         'skew_flux': pd.Series(test_flux).skew(),
#         'kurt_flux': pd.Series(test_flux).kurt()    
# }
# feature_list.append(features)

# features_df = pd.DataFrame(feature_list)


# pandas_dfs = []
# lc1_list = []

# for lc in lcs:
#     if lc is not None:  # Check if lc is a valid LightCurve object
#         pandas_df = lc.to_pandas()
#         if 'time' not in pandas_df.columns:
#             pandas_df = pandas_df.reset_index()
#         pandas_df = pandas_df.dropna(subset=['time', 'flux', 'sap_flux'])
#         pandas_df['flux'] = (pandas_df['flux'] - pandas_df['flux'].mean()) / pandas_df['flux'].std()
#         pandas_df['sap_flux'] = (pandas_df['sap_flux'] - pandas_df['sap_flux'].mean()) / pandas_df['sap_flux'].std()
#         pandas_dfs.append(pandas_df)
    

# rows = len(pandas_dfs)
# print(f"Shape of pandas dfs: {rows}")  

# cols = len(pandas_dfs[0]) if rows > 0 else 0
# print(f"Number of columns: {cols}")

# for i in range(1, rows) :
#     ## plotting tess sap_flux over time
#     plt.figure(figsize=(12, 5))
#     plt.plot(pandas_dfs[i]['time'], pandas_dfs[i]['flux'], label='flux', alpha=0.7)
#     plt.plot(pandas_dfs[i]['time'], pandas_dfs[i]['sap_flux'], label='sap_flux', alpha=0.7)
#     #plt.plot(pandas_dfs[i]['time'], pandas_dfs[i]['kspsap_flux'], label='kspsap_flux', alpha=0.7)
#     plt.xlabel('Time (BJD - 2457000)')
#     plt.ylabel('SAP Flux')
#     plt.title('TESS Light Curve: SAP Flux vs Time')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()