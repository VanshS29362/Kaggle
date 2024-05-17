# Comment Vansh Saxena
# Github Repository:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Separate target from predictors
y = train_data['SalePrice']
X = train_data.drop(['SalePrice'], axis=1)

# Split the data into training and validation datasets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Preprocessing for data
numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]
numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),('scaler', StandardScaler())])
categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == "object"]
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Feature engineering
X_train['OverallQual*OverallCond'] = X_train['OverallQual'] * X_train['OverallCond']
X_valid['OverallQual*OverallCond'] = X_valid['OverallQual'] * X_valid['OverallCond']
test_data['OverallQual*OverallCond'] = test_data['OverallQual'] * test_data['OverallCond']
X_train['TotalBath'] = X_train['FullBath'] + 0.5 * X_train['HalfBath']
X_valid['TotalBath'] = X_valid['FullBath'] + 0.5 * X_valid['HalfBath']
test_data['TotalBath'] = test_data['FullBath'] + 0.5 * test_data['HalfBath']

# Update numerical columns to include new features
numerical_cols += ['OverallQual*OverallCond', 'TotalBath']
models = {
    'DecisionTree': DecisionTreeRegressor(random_state=0),
    'RandomForest': RandomForestRegressor(random_state=0),
    'GradientBoosting': GradientBoostingRegressor(random_state=0)
}

def evaluate_model(model, X_train, X_valid, y_train, y_valid):
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    return rmse

for name, model in models.items():
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    score = evaluate_model(model_pipeline, X_train, X_valid, y_train, y_valid)
    print(f"{name} RMSE: {score}")

# Choose the best model based on validation performance
best_model = GradientBoostingRegressor(random_state=0)

# Create and train the final model pipeline
final_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', best_model)])
final_pipeline.fit(X, y)

# Make predictions on the test set
test_preds = final_pipeline.predict(test_data)
submission = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': test_preds})
submission.to_csv('submission.csv', index=False)


