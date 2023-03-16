# import the python libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# read dataset csv file
df = pd.read_csv('heart.csv')

# remove any null values from data rows
df = df.dropna()

# get the column names
column_names = df.columns

# using label encoder
le = LabelEncoder()

# save previous dataset
df.to_csv('prev_heart.csv', index=False)

# scan for columns with non-numerical labels
for column in column_names:
    # checks if the first label is string
    if isinstance(df[column][0], str):
        # transform to numerical labels
        df[column] = le.fit_transform(df[column])

# save refined dataset
df.to_csv('new_heart.csv', index=False)

# plot heatmap
fig = plt.figure(figsize=(15, 7))

# set the figure title
fig.canvas.manager.set_window_title('MSc Artificial Intelligent Task: Heat Map Plot By Umolu John Chukwuemeka')

# title by setting initial sizes
fig.suptitle('Heart Disease Heatmap Correlation Plot', fontsize=14, fontweight='bold')

# plot the heatmap
sns.heatmap(df.corr(), annot=True)

# add a space at the bottom of the plot
fig.subplots_adjust(bottom=0.2)


# display the plot
plt.show()