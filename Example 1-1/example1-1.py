# import modules
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import os 

# current working directory
cwd = os.path.dirname(os.path.realpath(__file__))

# load the data
oecd_bli = pd.read_csv(os.path.join(cwd, r'data\oecd_bli_2015.csv'), thousands=',')
gdp_per_capita = pd.read_csv(os.path.join(cwd, r'data\gdp_per_capita.csv'), thousands=',', delimiter='\t', encoding='latin1', na_values="n/a")

# Prepare the data & create dataframe 
def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

# slice dataframe for X and y
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# visualise the data
country_stats.plot(kind='scatter', x="GDP per capita", y="Life satisfaction")
plt.show()

# select a linear model (linear regression is MODEL-BASED LEARNING)
model = LinearRegression()

# train the model
model.fit(X, y)

# make a prediction for cyprus
X_new = [[22587]] # cyprus GDP per capita
cyprus_life_satisfaction_prediction = model.predict(X_new)
print("linear regression prediction for cyprus life satisfaction is: ", cyprus_life_satisfaction_prediction[0][0])

# visualize the prediction
plt.subplot2grid((1, 1), (0, 0))
plt.plot(X_new[0][0], cyprus_life_satisfaction_prediction[0][0], 'ro', label='cyprus - linear regression')
plt.scatter(X, y)
plt.legend()
plt.show()

# NEW MODEL
# ---------

# now lets try k-Nearest Neighbors (INSTANCE-BASED LEARNING)
model_2 = KNeighborsRegressor(n_neighbors=3)

# train the model
model_2.fit(X, y)

# make a prediction for cyprus
X_new = [[22587]] # cyprus GDP per capita
cyprus_life_satisfaction_prediction_K_nearest_neighbors = model_2.predict(X_new)
print("K nearest neighbors prediction for cyprus life satisfaction is: ", cyprus_life_satisfaction_prediction_K_nearest_neighbors[0][0])

# visualize the new prediction
plt.subplot2grid((1, 1), (0, 0))
plt.plot(X_new[0][0], cyprus_life_satisfaction_prediction[0][0], 'ro', label='cyprus - linear regression')
plt.plot(X_new[0][0], cyprus_life_satisfaction_prediction_K_nearest_neighbors[0][0], 'go', label='cyprus - k nearest neighbor')
plt.scatter(X, y)
plt.legend()
plt.show()