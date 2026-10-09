# Part 1: Importing Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import copy  # Importato per creare una copia dell'albero da potare
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

# Part 2: Loading Data
print("Loading data...")
pydata = pd.read_csv('data/pydata_cs_AGG.csv', parse_dates=['date_time'])
df_info = pd.read_csv('data/df_info_cs.csv')
print("Data loaded successfully.")

# Part 4: Visualizing Original Time Series
print("Visualizing original time series...")
pydata['date'] = pydata['date_time'].dt.date
pydata['time'] = pydata['date_time'].dt.time
pivot_original = pydata.pivot_table(index='date', columns='time', values='energy_h')
plt.figure(figsize=(15, 10))
sns.heatmap(pivot_original, cmap='Spectral_r', cbar_kws={'label': 'Electrical energy [kWh]'}, xticklabels=48,
            yticklabels=30)
plt.title('Carpet Plot of Energy Consumption (Original Data)')
plt.xlabel('Time of Day')
plt.ylabel('Date')
plt.show()
print("Original time series visualization complete.")

print("Sostituzione dei valori 0 con NaN...")
pydata['energy_h'] = pydata['energy_h'].replace(0, np.nan)

# Passaggio 2: Applica l'interpolazione lineare per riempire i NaN
print("Imputazione dei valori NaN (che erano 0) con interpolazione lineare...")
pydata['energy_h'] = pydata['energy_h'].interpolate(method='linear')

print("\nDati dopo l'interpolazione dei valori 0:")
print(pydata)
print("\nNumero di valori 0 dopo l'interpolazione:", (pydata['energy_h'] == 0).sum())
print("Numero di valori mancanti (NaN) dopo l'interpolazione:", pydata['energy_h'].isna().sum())

# Part 6: Visualizing Cleaned Time Series
print("Visualizing cleaned time series...")
pivot_cleaned = pydata.pivot_table(index='date', columns='time', values='energy_h')
plt.figure(figsize=(15, 10))
sns.heatmap(pivot_cleaned, cmap='Spectral_r', cbar_kws={'label': 'Electrical energy [kWh]'}, xticklabels=48,
            yticklabels=30, vmin=0)
plt.title('Carpet Plot of Energy Consumption (Cleaned Data)')
plt.xlabel('Time of Day')
plt.ylabel('Date')
plt.show()
print("Cleaned time series visualization complete.")

# Part 7: Constructing the M x N Matrix
print("Constructing the M x N matrix...")
py_mxn = pydata.pivot(index='date', columns='time', values='energy_h').reset_index()
print("M x N matrix constructed.")

# Part 8: Normalizing the Data
print("Normalizing the data...")
data_columns = py_mxn.columns[1:]
py_mxn[data_columns] = py_mxn[data_columns].div(py_mxn[data_columns].max(axis=1), axis=0)
print("Data normalization complete.")

# Part 9: Computing the Distance Matrix
print("Computing the distance matrix...")
distance_matrix = pdist(py_mxn[data_columns], metric='euclidean')
distance_square_matrix = squareform(distance_matrix)
print("Distance matrix computed.")

# Part 10: Hierarchical Clustering
print("Performing hierarchical clustering...")
linked = linkage(distance_matrix, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, labels=py_mxn['date'].astype(str).values)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Date')
plt.ylabel('Distance')
plt.show()
print("Hierarchical clustering complete.")

# Part 11: Determining Optimal Number of Clusters
print("Determining optimal number of clusters...")
from sklearn.metrics import silhouette_score

range_n_clusters = range(3, 16)
silhouette_avg = []

for n_clusters in range_n_clusters:
    print(f"Evaluating silhouette score for {n_clusters} clusters...")
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = clusterer.fit_predict(py_mxn[data_columns])
    silhouette_avg.append(silhouette_score(py_mxn[data_columns], cluster_labels))

plt.figure(figsize=(8, 5))
plt.plot(range_n_clusters, silhouette_avg, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Average Silhouette Score')
plt.title('Silhouette Analysis For Optimal k')
plt.show()
print("Optimal number of clusters determined.")

# Part 12: Assigning Cluster Labels
n_clusters = 3
print(f"Assigning cluster labels for {n_clusters} clusters...")
cluster_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
py_mxn['cluster'] = cluster_model.fit_predict(py_mxn[data_columns])

plt.figure(figsize=(10, 7))
dendrogram(linked, labels=py_mxn['date'].astype(str).values, color_threshold=linked[-(n_clusters - 1), 2])
plt.title('Dendrogram with Cluster Cuts')
plt.xlabel('Date')
plt.ylabel('Distance')
plt.axhline(y=linked[-(n_clusters - 1), 2], c='k', ls='--', lw=0.5)
plt.show()
print("Cluster labels assigned.")

# =============================================================================
# Part 13: Calculating Cluster Centroids and Errors (MODIFICATO E OTTIMIZZATO)
# =============================================================================
print("Calculating cluster centroids and errors...")
melted_py_mxn = py_mxn.melt(id_vars=['date', 'cluster'], var_name='time', value_name='norm_energy_h')

# --- MODIFICA CHIAVE: Convertiamo la colonna 'time' una sola volta ---
# Convertiamo l'oggetto 'time' in una stringa e poi in un oggetto datetime completo.
# Specificando il formato '%H:%M:%S', eliminiamo l'avviso e rendiamo l'operazione più veloce.
melted_py_mxn['time'] = pd.to_datetime(melted_py_mxn['time'].astype(str), format='%H:%M:%S')

# Calcolo dei centroidi (media e dev. standard)
centroids = melted_py_mxn.groupby(['cluster', 'time']).agg(
    avg_var=('norm_energy_h', 'mean'),
    sd_var=('norm_energy_h', 'std')
).reset_index()
print("Cluster centroids calculated.")

# Calcolo dell'errore (RMSE) per ogni cluster
cluster_errors = {}
for i in range(n_clusters):
    cluster_data = py_mxn[py_mxn['cluster'] == i]
    # Ora il centroide ha già una colonna 'time' di tipo datetime, quindi non la usiamo qui
    centroid_profile = centroids[centroids['cluster'] == i]['avg_var'].values
    profiles_in_cluster = cluster_data[data_columns].values

    squared_errors = np.mean((profiles_in_cluster - centroid_profile) ** 2, axis=1)
    rmse = np.sqrt(np.mean(squared_errors))
    cluster_errors[i] = rmse
    print(f"Cluster {i} - Average RMSE vs Centroid: {rmse:.4f}")

# =============================================================================
# Part 14: Visualizing Clusters and Centroids (SEMPLIFICATO)
# =============================================================================
print("Visualizing clusters and centroids...")


def facet_plot(data, color):
    cluster = data['cluster'].iloc[0]

    # 1. Disegna i profili giornalieri. La colonna 'time' è già nel formato corretto!
    for date, group in data.groupby('date'):
        plt.plot(group['time'], group['norm_energy_h'], color='grey', alpha=0.3, zorder=1)

    # 2. Dati del centroide. Anche qui, 'time' è già pronto.
    centroid_data = centroids[centroids['cluster'] == cluster]

    # 3. Disegna l'area di deviazione standard
    plt.fill_between(centroid_data['time'],
                     centroid_data['avg_var'] - centroid_data['sd_var'],
                     centroid_data['avg_var'] + centroid_data['sd_var'],
                     color='red', alpha=0.2, zorder=2)

    # 4. Disegna la linea del centroide
    plt.plot(centroid_data['time'], centroid_data['avg_var'], color='red', linestyle='--', linewidth=2, zorder=3)

    # Mostra l'RMSE calcolato
    rmse_val = cluster_errors[cluster]
    plt.text(0.05, 0.95, f'RMSE: {rmse_val:.3f}', transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))

    plt.title(f'Cluster {cluster}')
    plt.xlabel('Time')
    plt.ylabel('Normalized Energy')
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.ylim(0, 1.1)
    plt.tight_layout()


g = sns.FacetGrid(melted_py_mxn, col='cluster', col_wrap=2, height=4, sharey=True)
g.map_dataframe(facet_plot)
plt.show()
print("Cluster and centroid visualization complete.")

# Part 15: Classification Tree
print("Preparing data for classification tree...")
df_info['day'] = df_info['day'].str[:3]
py_mxn['date'] = pd.to_datetime(py_mxn['date'])
df_info['date'] = pd.to_datetime(df_info['date'])
classification_data = pd.merge(py_mxn, df_info)

X = classification_data[['T_avg', 'T_max', 'T_min', 'day']]
y = classification_data['cluster']
X = pd.get_dummies(X, columns=['day'], drop_first=True)

print("Training classification tree...")
clf = DecisionTreeClassifier(min_samples_leaf=10)
clf.fit(X, y)
print("Classification tree trained.")

# Part 16: Plotting the Decision Tree
print("Plotting the decision tree...")
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=[str(i) for i in clf.classes_], filled=True)
plt.title("Full Decision Tree")
plt.show()
print("Decision tree plot complete.")

# =============================================================================
# Part 16.5: Pruning and Plotting the Simplified Tree (NUOVA SEZIONE)
# =============================================================================
print("Pruning redundant splits from the decision tree...")


def prune_redundant_splits(clf):
    tree = clf.tree_

    def find_majority_class(node_id):
        return np.argmax(tree.value[node_id])

    for i in range(tree.node_count):
        if tree.children_left[i] != tree.children_right[i]:
            left_child = tree.children_left[i]
            right_child = tree.children_right[i]
            left_class = find_majority_class(left_child)
            right_class = find_majority_class(right_child)

            if left_class == right_class:
                tree.children_left[i] = -1
                tree.children_right[i] = -1
    return clf


# Crea una copia profonda dell'albero originale per non modificarlo
clf_pruned = copy.deepcopy(clf)
clf_pruned = prune_redundant_splits(clf_pruned)

print("Plotting the pruned decision tree...")
plt.figure(figsize=(15, 8))
plot_tree(clf_pruned,
          feature_names=X.columns,
          class_names=[str(i) for i in clf.classes_],
          filled=True,
          impurity=False,
          proportion=True,
          fontsize=10)
plt.title("Pruned Decision Tree (Redundant Splits Removed)")
plt.show()
print("Pruned decision tree plot complete.")

# Part 17: Model Evaluation
print("Evaluating model...")
classification_data['pred'] = clf.predict(X)
conf_matrix = confusion_matrix(classification_data['cluster'], classification_data['pred'])
accuracy = accuracy_score(classification_data['cluster'], classification_data['pred']) * 100
recall = recall_score(classification_data['cluster'], classification_data['pred'], average='macro') * 100
precision = precision_score(classification_data['cluster'], classification_data['pred'], average='macro') * 100

print("Confusion Matrix:\n", conf_matrix)
print(f"Accuracy: {accuracy:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"Precision: {precision:.2f}%")
print("Model evaluation complete.")