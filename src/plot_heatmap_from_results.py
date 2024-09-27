import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
sns.set_theme()

# Load the example flights dataset and convert to long-form
# flights_long = sns.load_dataset("flights")
# flights = (
#     flights_long
#     .pivot(index="month", columns="year", values="passengers")
# )

# import os
# import glob

# def find_files(directory, pattern):
#     """Recursively finds all files in the given directory matching the given pattern."""
#     file_paths = []
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if os.path.exists(os.path.join(root, file)):
#                 if glob.glob(os.path.join(root, file), recursive=True) and pattern in file:
#                     file_paths.append(os.path.abspath(os.path.join(root, file)))
#     return file_paths

# folder_path = "/Users/trex22/Downloads/ClassificationV2/Random"
# pattern = 'total_result_grouped_percetages.csv'

# file_paths = find_files(folder_path, pattern)
# print(file_paths)

# data = pd.read_csv('/Users/trex22/repos/masters/dissertation/data/rng_binary_avg_results.csv')
# pivot_table = data.transpose()
# melted_data = data.reset_index().melt(id_vars='group', var_name='variable', value_name='value')

# data = data.set_index('group')
# data.index.name = None
# groups = ['flat', 'human', 'vehicle', 'construction', 'object', 'nature', 'sky', 'void']
# pivot_data = data.melt(id_vars='group', value_vars=groups).rename(columns={'variable': ''})
# pivot_data.columns = pivot_data.columns.str.replace('', '_')

# pivot_data.index = pivot_data['group']
# pivot_data.index.name = None
# pivot_data = pivot_data.reset_index()

# sns.set_style("whitegrid")
# pivot_table = data.transpose()

# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(10, 8))

# sns.heatmap(table, annot=True, fmt="d", linewidths=.5, ax=ax)
# heatmap = sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap='RdYlGn')
# heatmap = sns.heatmap(data=pivot_data, cmap="coolwarm", annot=True, fmt='.2f', xticklabels=pivot_data.columns, yticklabels=pivot_data.columns)
# heatmap = sns.heatmap(df, cmap="coolwarm", annot=True, fmt='.2f')

data = pd.read_csv('/Users/trex22/repos/masters/dissertation/data/rng_binary_avg_results.csv')
# groups = ['flat', 'human', 'vehicle', 'construction', 'object', 'nature', 'sky', 'void']

labels = ['bicycle', 'dynamic', 'guard rail', 'rail track', 'rider', 'sky', 'terrain', 'traffic light']
groups = ['vehicle', 'void', 'construction', 'flat', 'human', 'sky', 'nature', 'object']

# labels = ['guard rail', 'rail track', 'rider', 'terrain', 'traffic light', 'sky', 'bicycle', 'dynamic']
# groups = ['construction', 'flat', 'human', 'nature', 'object', 'sky', 'vehicle', 'void']

# pivot_data = data.melt(id_vars='group', value_vars=groups).rename(columns={'variable': ''})
# pivot_data.index = pivot_data['group']
# pivot_data.index.name = None

# np_data = np.array(pivot_data)
# # np_data = np_data[np_data[:,0].argsort(), :]
# # np_data = np_data[np_data[:,1].argsort(kind='mergesort')]

# np_data = np_data[np_data[:,1].argsort(), :]
# np_data = np_data[np_data[:,0].argsort(kind='mergesort')]

# row_labels = ["flat", "human", "vehicle", "construction", "object", "nature", "sky", "void"]
# # np_data = np_data[np.array([row_labels.index(row) for row in np_data[:,0]]), :]

# df = pd.DataFrame(np_data, columns=['group_x', 'group_y', 'percentage'])

# df['group_x'] = df['group_x'].astype(str)
# df['group_y'] = df['group_y'].astype(str)
# df['percentage'] = df['percentage'].astype(float)
# df = df.set_index(['group_x', 'group_y'])

# summary = pd.pivot_table(data=df, index='group_x', columns='group_y', values='percentage')

# # cmap="coolwarm"
# heatmap = sns.heatmap(summary, annot=True, fmt='.2f')

# # heatmap.set_xlabel("Groups")
# # heatmap.set_ylabel("Groups")
# # heatmap.set_title("Heat-map of GradSUM Average Results for 100 Random Models Performing Binary Classification")
# heatmap.set_xlabel("")
# heatmap.set_ylabel("")

# plt.show()
# # fig = swarm_plot.get_figure()
# f.savefig("rng_bin_models_heatmap.png")

# Sorted List
labels = ['dynamic', 'sky', 'terrain', 'traffic light', 'guard rail', 'bicycle', 'rider', 'rail track']
groups = ['flat', 'human', 'vehicle', 'construction', 'object', 'nature', 'sky', 'void']
# groups = ['object', '']

# Manual data processing
np_data = []

# Index(['group', 'flat', 'human', 'vehicle', 'construction', 'object', 'nature', 'sky', 'void'],

for i in range(len(data)):
  row = data.iloc[i]

  for group in groups:
    np_data.append([row["group"], group, row[group]])

df = pd.DataFrame(np_data, columns=['Labels', 'Groups', 'percentage'])

df['Labels'] = df['Labels'].astype(str)
df['Groups'] = df['Groups'].astype(str)
df['percentage'] = df['percentage'].astype(float)
df = df.set_index(['Groups', 'Labels'])

summary = pd.pivot_table(data=df, index='Labels', columns='Groups', values='percentage')
summary = pd.DataFrame(summary)[groups]
summary = summary.reindex(labels, level=0)
# summary.sort_values(by="Labels", categories=labels)
# summary.Categorical(labels, ordered=True, categories=labels)

heatmap = sns.heatmap(summary, annot=True, fmt='.2f')

# heatmap.set_xlabel("")
# heatmap.set_ylabel("")

plt.show()
f.savefig("rng_bin_models_heatmap.png")
