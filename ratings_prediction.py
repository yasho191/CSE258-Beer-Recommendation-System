import pandas as pd
import json
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv('ratebeer_subset.csv')
beer_style_encoder = LabelEncoder()
df['beer/style'] = beer_style_encoder.fit_transform(df['beer/style'])
df["review/time"] = pd.to_datetime(df["review/time"])
df["review_month"] = df["review/time"].dt.month

allRatings = []
for index, row in df.iterrows():
	user = row['review/profileName']
	item = row['beer/name']
	rating = row['review/overall']
	allRatings.append((user, item, rating))
 
ratingsTrain, ratingsValid = train_test_split(allRatings, test_size=0.2, random_state=42)

print(f"Training set size: {len(ratingsTrain)}")
print(f"Validation set size: {len(ratingsValid)}")

MSE = []
train_MSE = []
R2 = []
train_R2 = []
models = [
    "LFM",
	"Linear Regression", 
	"Lasso", 
	"Ridge", 
	"Decision Tree", 
	"Random Forest", 
	"Neural Network", 
	"Baseline Mean",
	"Baseline Median"
]

class RatingPredictionModelLFM:
	def __init__(self, mu, lamb):
		self.lamb = lamb
		self.alpha = mu
		self.beta_u = defaultdict(float)
		self.beta_i = defaultdict(float)
		self.ratingsPerUser = defaultdict(list)
		self.ratingsPerItem = defaultdict(list)

	def fit(self, ratingsTrain, ratingsValid, epochs=20):
		for u, i, r in ratingsTrain:
			self.ratingsPerUser[u].append((i, r))
			self.ratingsPerItem[i].append((u, r))

		# Iterative updates for biases
		for epoch in range(epochs):

			for u in self.ratingsPerUser:
				numerator = sum([r - self.alpha - self.beta_i[i] for i, r in self.ratingsPerUser[u]])
				denominator = self.lamb + len(self.ratingsPerUser[u])
				self.beta_u[u] = numerator / denominator

			for i in self.ratingsPerItem:
				numerator = sum([r - self.alpha - self.beta_u[u] for u, r in self.ratingsPerItem[i]])
				denominator = self.lamb + len(self.ratingsPerItem[i])
				self.beta_i[i] = numerator / denominator

			numerator = sum([r - self.beta_u[u] - self.beta_i[i] for u, i, r in ratingsTrain])
			self.alpha = numerator / len(ratingsTrain)

		loss = self.evaluate(ratingsValid)
		print(f"Final Validation Loss = {loss}")
		return loss

	def predict(self, user, item):
		bu = self.beta_u.get(user, 0)
		bi = self.beta_i.get(item, 0)
		return self.alpha + bu + bi

	def evaluate(self, ratingsValid):
		mse_valid = 0
		for u, i, r in ratingsValid:
			prediction = self.predict(u, i)
			mse_valid += (r - prediction) ** 2
		mse_valid /= len(ratingsValid)
		return mse_valid


# lam = best_lambda
lam = 1.3
mu = np.mean([r for _, _, r in ratingsTrain])
rating_model = RatingPredictionModelLFM(mu, lam)
print("Lambda: ", lam)
mse = rating_model.fit(ratingsTrain, ratingsValid, epochs=50)
MSE.append(mse)

predictions = [rating_model.predict(u, i) for u, i, _ in ratingsValid]
r2 = r2_score([r for _, _, r in ratingsValid], predictions)
R2.append(r2)

mse_train = 0
for u, i, r in ratingsTrain:
	prediction = rating_model.predict(u, i)
	mse_train += (r - prediction) ** 2
train_MSE.append(mse_train / len(ratingsTrain))

predictions = [rating_model.predict(u, i) for u, i, _ in ratingsTrain]
r2_train = r2_score([r for _, _, r in ratingsTrain], predictions)
train_R2.append(r2_train)

print("Training Score")
print("LFM MSE: ", mse_train / len(ratingsTrain))
print("LFM R2: ", r2_train / len(ratingsTrain))
print("Validation Score")
print("LFM MSE: ", mse)
print("LFM R2: ", r2 / len(ratingsValid))

df.drop(columns=["review/text", "review/time", "review/profileName", "beer/name", "beer/beerId", "beer/brewerId", "beer/style", "review_length", "review_month"], inplace=True)
y = df["review/overall"].values
df.drop(columns=["review/overall"], inplace=True)

scaler = StandardScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df), columns=df.columns)

X = df.iloc[:, :].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
MSE.append(mse)
r2 = r2_score(y_test, y_pred)
R2.append(r2)

y_train_pred = linear_model.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
train_MSE.append(mse_train)
r2_train = r2_score(y_train, y_train_pred)
train_R2.append(r2_train)

print("Training Score")
print("Linear Regression MSE: ", mse_train)
print("Linear Regression R2: ", r2_train)
print("Validation Score")
print("Linear Regression MSE: ", mse)
print("Linear Regression R2: ", r2)

feature_importance = linear_model.coef_
features = list(df.columns)
print("Feature Importance: ", {features[i]: feature_importance[i] for i in range(len(features))})

lasso_model = Lasso()
lasso_model.fit(X_train, y_train)
y_pred = lasso_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
MSE.append(mse)
r2 = r2_score(y_test, y_pred)
R2.append(r2)

y_train_pred = lasso_model.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
train_MSE.append(mse_train)
r2_train = r2_score(y_train, y_train_pred)
train_R2.append(r2_train)

print("Training Score")
print("Lasso MSE: ", mse_train)
print("Lasso R2: ", r2_train)
print("Validation Score")
print("Lasso MSE: ", mse)
print("Lasso R2: ", r2)

feature_importance = lasso_model.coef_
features = list(df.columns)
print("Feature Importance: ", {features[i]: feature_importance[i] for i in range(len(features))})

ridge_model = Ridge()
ridge_model.fit(X_train, y_train)
y_pred = ridge_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
MSE.append(mse)
r2 = r2_score(y_test, y_pred)
R2.append(r2)

y_train_pred = ridge_model.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
train_MSE.append(mse_train)
r2_train = r2_score(y_train, y_train_pred)
train_R2.append(r2_train)

print("Training Score")
print("Ridge MSE: ", mse_train)
print("Ridge R2: ", r2_train)
print("Validation Score")
print("Ridge MSE: ", mse)
print("Ridge R2: ", r2)

feature_importance = ridge_model.coef_
features = list(df.columns)
print("Feature Importance: ", {features[i]: feature_importance[i] for i in range(len(features))})

decision_tree = DecisionTreeRegressor(random_state=42, max_depth=10)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
MSE.append(mse)
r2 = r2_score(y_test, y_pred)
R2.append(r2)

y_train_pred = decision_tree.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
train_MSE.append(mse_train)
r2_train = r2_score(y_train, y_train_pred)
train_R2.append(r2_train)

print("Training Score")
print("Decision Tree MSE: ", mse_train)
print("Decision Tree R2: ", r2_train)
print("Validation Score")
print("Decision Tree MSE: ", mse)
print("Decision Tree R2: ", r2)

feature_importance = decision_tree.feature_importances_
features = list(df.columns)
print("Feature Importance: ", {features[i]: feature_importance[i] for i in range(len(features))})

random_forest = RandomForestRegressor(n_estimators=15, random_state=42)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
MSE.append(mse)
r2 = r2_score(y_test, y_pred)
R2.append(r2)

y_train_pred = random_forest.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
train_MSE.append(mse_train)
r2_train = r2_score(y_train, y_train_pred)
train_R2.append(r2_train)

print("Training Score")
print("Random Forest MSE: ", mse_train)
print("Random Forest R2: ", r2_train)
print("Validation Score")
print("Random Forest MSE: ", mse)
print("Random Forest R2: ", r2)

feature_importance = random_forest.feature_importances_
features = list(df.columns)
print("Feature Importance: ", {features[i]: feature_importance[i] for i in range(len(features))})

mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=20, random_state=42)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
MSE.append(mse)
r2 = r2_score(y_test, y_pred)
R2.append(r2)

y_train_pred = mlp.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
train_MSE.append(mse_train)
r2_train = r2_score(y_train, y_train_pred)
train_R2.append(r2_train)

print("Training Score")
print("Neural Network MSE: ", mse_train)
print("Neural Network R2: ", r2_train)
print("Validation Score")
print("Neural Network MSE: ", mse)
print("Neural Network R2: ", r2)


# Baseline Model
baseline_mse = np.mean((y_test - np.mean(y_train)) ** 2)
MSE.append(baseline_mse)
baseline_r2 = r2_score(y_test, [np.mean(y_train) for _ in range(len(y_test))])
R2.append(baseline_r2)
train_baseline_mse = np.mean((y_train - np.mean(y_train)) ** 2)
train_MSE.append(train_baseline_mse)
train_baseline_r2 = r2_score(y_train, [np.mean(y_train) for _ in range(len(y_train))])
train_R2.append(train_baseline_r2)

print("Training Score")
print("Baseline MSE: ", train_baseline_mse)
print("Baseline R2: ", train_baseline_r2)
print("Validation Score")
print("Baseline MSE: ", baseline_mse)
print("Baseline R2: ", baseline_r2)

# Baseline Model Median
baseline_median_mse = np.mean((y_test - np.median(y_train)) ** 2)
MSE.append(baseline_median_mse)
predictor = np.median(y_train)
baseline_median_r2 = r2_score(y_test, [predictor for _ in range(len(y_test))])
R2.append(baseline_median_r2)
train_baseline_median_mse = np.mean((y_train - np.median(y_train)) ** 2)
train_MSE.append(train_baseline_median_mse)
predictor = np.median(y_train)
train_baseline_median_r2 = r2_score(y_train, [predictor for _ in range(len(y_train))])
train_R2.append(train_baseline_median_r2)

print("Training Score")
print("Baseline MSE: ", train_baseline_median_mse)
print("Baseline R2: ", train_baseline_median_mse)
print("Validation Score")
print("Baseline Median MSE: ", baseline_median_mse)
print("Baseline Median R2: ", baseline_median_r2)



results = {
	"Validation MSE": MSE,
	"Training MSE": train_MSE,
}

x = np.arange(len(models))  # the label locations
width = 0.35  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in results.items():
    offset = width * multiplier + 0.25
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    # ax.bar_label(rects, padding=3, fmt='%.4f')
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('MSE')
ax.set_title('MSE Comparison for Different Models')
ax.set_xticks(x + width, models, rotation=45)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 15)

plt.savefig("MSE.png")

results = {
    "Validation R2": R2,
	"Training R2": train_R2
}

x = np.arange(len(models))  # the label locations
width = 0.35  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in results.items():
    offset = width * multiplier + 0.25
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    # ax.bar_label(rects, padding=3, fmt='%.4f')
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('R2')
ax.set_title('R2 Score Comparison for Different Models')
ax.set_xticks(x + width, models, rotation=45)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 1)

plt.savefig("R2.png")

print("Best Model: ", models[np.argmin(MSE)])
print("Best Model MSE: ", np.min(MSE))

for i in range(len(models)):
	print(f"Model: {models[i]}")
	print(f"Training MSE: {train_MSE[i]}")
	print(f"Validation MSE: {MSE[i]}")
	print(f"Training R2: {train_R2[i]}")
	print(f"Validation R2: {R2[i]}")
	print()