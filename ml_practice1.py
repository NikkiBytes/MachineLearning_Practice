from sklearn import datasets
boston = datasets.load_boston()

# Shuffle the data
from sklearn.utils import shuffle
data, target = shuffle(boston.data, boston.target, random_state=0)

from matplotlib import pyplot as plt

for feature, name in zip(data.T, boston.feature_names):
    plt.figure(figsize=(4, 3))
    plt.scatter(feature, target)
    plt.xlabel(name, size=22)
    plt.ylabel('Price (US$)', size=22)
    plt.tight_layout()
    plt.show()


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()

from sklearn.model_selection import cross_val_score
print(cross_val_score(regressor, data, target))

print(cross_val_score(regressor, data[data[:, 3] == 0],
                      target[data[:, 3] == 0]))

print(cross_val_score(regressor, data[data[:, 3] == 1],
                      target[data[:, 3] == 1]))

print(cross_val_score(regressor, data[data[:, 3] == 0],
                      target[data[:, 3] == 0],
                      scoring='neg_mean_squared_error'))

print(cross_val_score(regressor, data[data[:, 3] == 1],
                      target[data[:, 3] == 1],
                      scoring='neg_mean_squared_error'))
