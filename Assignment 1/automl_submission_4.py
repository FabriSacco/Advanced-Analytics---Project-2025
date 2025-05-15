import pandas as pd
import numpy as np
from flaml import AutoML
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree, NearestNeighbors
from sklearn.covariance import EmpiricalCovariance
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import PowerTransformer, RobustScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import optuna
from optuna.integration import LightGBMPruningCallback
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_iterative_imputer  # needed for IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import haversine_distances
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import TargetEncoder
import os
import shap
from sklearn.inspection import permutation_importance
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Load datasets
train = pd.read_csv("Data/train.csv")
test = pd.read_csv("Data/test.csv")
stations = pd.read_csv("Data/stations.csv")

# Store target and IDs immediately after loading
train_ids = train["id"].copy()
test_ids = test["id"].copy()
y = train["price"].copy()  # Store target variable early

# Check what columns are available in the dataset
print(f"Train columns before removal: {train.columns.tolist()}")
print(f"Test columns before removal: {test.columns.tolist()}")

# Remove specified features if they exist
features_to_remove = ['added_time', 'is_promoted', 'sticker', 'price_drop_date', 'postcode']
removed_features = []

for feature in features_to_remove:
    if feature in train.columns:
        train = train.drop(columns=[feature])
        removed_features.append(feature)
    if feature in test.columns:
        test = test.drop(columns=[feature])

if removed_features:
    print(f"Removed features: {removed_features}")
else:
    print("None of the specified features found in the dataset")

# Verify the features were removed
for feature in features_to_remove:
    if feature in train.columns:
        print(f"WARNING: {feature} is still in train dataset!")
    if feature in test.columns:
        print(f"WARNING: {feature} is still in test dataset!")

print(f"Train columns after removal: {train.columns.tolist()}")
print(f"Test columns after removal: {test.columns.tolist()}")

# --- Domain-specific feature engineering ---

# Filter Belgian stations
stations = stations[stations["country-code"] == "be"][["latitude", "longitude"]]
station_coords_rad = np.radians(stations.to_numpy())
tree = BallTree(station_coords_rad, metric="haversine", leaf_size=40)

# Step 1: Initial simple imputation for numeric features
numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['id', 'price']]

# Separate coordinate columns for special handling
coord_cols = ['lat', 'lon']
other_numeric_cols = [col for col in numeric_cols if col not in coord_cols]

print("Performing imputation...")

# Print feature statistics
print("\nFeature statistics before imputation:")
for col in numeric_cols:
    stats = train[col].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
    missing = train[col].isnull().sum()
    print(f"\n{col}:")
    print(f"Missing values: {missing} ({(missing/len(train)*100):.1f}%)")
    print(f"Stats: {stats.to_dict()}")

class HaversineKNNImputer(BaseEstimator, TransformerMixin):
    """
    Custom KNN imputer using BallTree with haversine metric for efficient neighbor finding.
    Specifically designed for geographic (lat, lon) coordinate imputation.
    """
    def __init__(self, n_neighbors=3, weights="uniform"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        
    def fit(self, X, y=None):
        """
        Fit the imputer on X.
        X should be a 2D array with latitude and longitude columns.
        """
        # Store feature names
        self.feature_names_in_ = X.columns if hasattr(X, 'columns') else None
        
        # Convert to numpy array if DataFrame
        X_array = X.to_numpy() if hasattr(X, 'to_numpy') else np.asarray(X)
        
        if X_array.shape[1] != 2:
            raise ValueError("Input should have exactly 2 columns: [latitude, longitude]")
        
        # Store complete samples for later use
        self.complete_samples_mask_ = ~np.isnan(X_array).any(axis=1)
        self.complete_samples_ = X_array[self.complete_samples_mask_]
        
        if len(self.complete_samples_) < self.n_neighbors:
            raise ValueError(
                f"Found only {len(self.complete_samples_)} complete samples, "
                f"need at least {self.n_neighbors} for imputation."
            )
        
        # Initialize BallTree with complete samples
        self.tree_ = BallTree(
            np.radians(self.complete_samples_),
            metric='haversine'
        )
        
        return self
    
    def transform(self, X):
        """
        Impute missing values in X using fitted tree.
        """
        # Convert to numpy array if DataFrame
        X_array = X.to_numpy() if hasattr(X, 'to_numpy') else np.asarray(X)
        
        if X_array.shape[1] != 2:
            raise ValueError("Input should have exactly 2 columns: [latitude, longitude]")
            
        X_imputed = X_array.copy()
        missing_rows = np.isnan(X_array).any(axis=1)
        
        if not np.any(missing_rows):
            return pd.DataFrame(X_imputed, columns=self.feature_names_in_, index=X.index if hasattr(X, 'index') else None)
            
        missing_idx = np.where(missing_rows)[0]
        for idx in missing_idx:
            row = X_array[idx]
            missing_mask = np.isnan(row)
            
            if missing_mask.all():
                X_imputed[idx] = np.mean(self.complete_samples_, axis=0)
                continue
            
            query = np.radians(row.reshape(1, -1))
            distances, indices = self.tree_.query(
                query,
                k=self.n_neighbors,
                return_distance=True
            )
            
            distances = distances.ravel() * 6371.0
            
            if self.weights == "uniform":
                weights = np.ones_like(distances)
            else:
                weights = 1 / (distances + np.finfo(float).eps)
            
            weights /= np.sum(weights)
            
            neighbor_values = self.complete_samples_[indices.ravel()]
            X_imputed[idx, missing_mask] = np.average(
                neighbor_values[:, missing_mask],
                weights=weights,
                axis=0
            )
        
        return pd.DataFrame(X_imputed, columns=self.feature_names_in_, index=X.index if hasattr(X, 'index') else None)

    def get_feature_names_out(self, feature_names_in=None):
        """Return feature names for output features.
        
        Parameters
        ----------
        feature_names_in : array-like of str or None, default=None
            Input feature names. If None, then the feature names from fit are used.
            
        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        if feature_names_in is None:
            if self.feature_names_in_ is not None:
                return np.asarray(self.feature_names_in_)
            return np.array([f"feature{i}" for i in range(2)])
        return np.asarray(feature_names_in)

# Separate columns by their characteristics
energy_cols = ['energy_value']
area_cols = ['area']
bedrooms_cols = ['bedrooms']  # Special handling for discrete count data
foto_cols = ['foto_amount']   # Another count feature
binary_cols = ['is_appartment', 'new_building']
categorical_cols = ['advertiser', 'subtype', 'energy_label', 'province']
other_numeric_cols = [col for col in other_numeric_cols 
                     if col not in energy_cols + area_cols + bedrooms_cols + foto_cols + binary_cols]

# Create custom transformers with proper feature name handling
def clip_outliers(X):
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(
            np.clip(X.values, 0, np.percentile(X.values[~np.isnan(X.values)], 99)),
            index=X.index,
            columns=X.columns
        )
    return np.clip(X, 0, np.percentile(X[~np.isnan(X)], 99))

def log_transform(X):
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(
            np.log1p(np.maximum(X.values, 0)),
            index=X.index,
            columns=X.columns
        )
    return np.log1p(np.maximum(X, 0))

# Create the pipelines with pandas output
coords_pipe = Pipeline([
    ("imputer", HaversineKNNImputer(n_neighbors=3, weights="distance"))
]).set_output(transform="pandas")

energy_pipe = Pipeline([
    ("imputer", KNNImputer(n_neighbors=5)),
    ("clip", FunctionTransformer(
        lambda x: np.clip(x, 0, np.percentile(x[~np.isnan(x)], 99.5)),
        feature_names_out="one-to-one"
    )),
    ("log", FunctionTransformer(
        lambda x: np.log1p(x),
        feature_names_out="one-to-one"
    )),
    ("scaler", RobustScaler())
]).set_output(transform="pandas")

area_pipe = Pipeline([
    ("imputer", KNNImputer(n_neighbors=5)),
    ("clip", FunctionTransformer(
        lambda x: np.clip(x, 0, np.percentile(x[~np.isnan(x)], 99.5)),
        feature_names_out="one-to-one"
    )),
    ("log", FunctionTransformer(
        lambda x: np.log1p(np.maximum(x, 0)),
        feature_names_out="one-to-one"
    )),
    ("scaler", RobustScaler())
]).set_output(transform="pandas")

# Special pipeline for count data (bedrooms, foto_amount)
count_pipe = Pipeline([
    ("imputer", KNNImputer(n_neighbors=3)),
    ("clip", FunctionTransformer(
        lambda x: np.clip(x, 0, np.percentile(x[~np.isnan(x)], 99)),
        feature_names_out="one-to-one"
    )),
    ("round", FunctionTransformer(
        lambda x: np.round(x),
        feature_names_out="one-to-one"
    )),
    ("scaler", RobustScaler())
]).set_output(transform="pandas")

# Binary features pipeline
binary_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy='most_frequent')),  # Use mode for binary features
    ("binarizer", FunctionTransformer(
        lambda x: x.astype(float),
        feature_names_out="one-to-one"
    ))
]).set_output(transform="pandas")

# Categorical features pipeline with target encoding
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy='constant', fill_value='missing')),
    ("encoder", TargetEncoder(
        target_type='continuous',
        smooth='auto',
        cv=5,  # Use 5-fold cross-validation
        random_state=42
    ))
]).set_output(transform="pandas")

numeric_pipe = Pipeline([
    ("scaler", RobustScaler()),
    ("imputer", KNNImputer(n_neighbors=3))
]).set_output(transform="pandas")

# Create separate preprocessors for features that don't need target and those that do
# First preprocessor: everything except categorical features
main_preprocessor = ColumnTransformer(
    transformers=[
        ("coords", coords_pipe, coord_cols),
        ("energy", energy_pipe, energy_cols),
        ("area", area_pipe, area_cols),
        ("bedrooms", count_pipe, bedrooms_cols),
        ("fotos", count_pipe, foto_cols),
        ("binary", binary_pipe, binary_cols),
        ("numeric", numeric_pipe, other_numeric_cols)
    ],
    verbose_feature_names_out=False
).set_output(transform="pandas")

# Second preprocessor: just for categorical features with target encoding
categorical_preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_pipe, categorical_cols)
    ],
    verbose_feature_names_out=False
).set_output(transform="pandas")

# Fit and transform main preprocessor (no target needed)
print("Fitting main preprocessor...")
main_preprocessor.fit(train)
train_transformed_main = main_preprocessor.transform(train)
test_transformed_main = main_preprocessor.transform(test)

# Fit and transform categorical features with cross-validated target encoding
print("Fitting categorical preprocessor with cross-validated target encoding...")
categorical_preprocessor.fit(train[categorical_cols], train['price'])
# The transform will use cross-validated predictions for training data
train_transformed_cat = categorical_preprocessor.transform(train[categorical_cols])
test_transformed_cat = categorical_preprocessor.transform(test[categorical_cols])

# Combine the transformed features
train_transformed = pd.concat([train_transformed_main, train_transformed_cat], axis=1)
test_transformed = pd.concat([test_transformed_main, test_transformed_cat], axis=1)

# Instead of assigning back to original dataframes, use the transformed data directly
print("Successfully completed preprocessing with cross-validated target encoding")

# Keep the transformed data for further processing
train = train_transformed
test = test_transformed

# Step 2: Distance to nearest major Belgian city
cities = {
    "Brussels": (50.8503, 4.3517), "Antwerp": (51.2194, 4.4025),
    "Ghent": (51.0543, 3.7174), "Charleroi": (50.4114, 4.4446),
    "Liège": (50.6326, 5.5797), "Bruges": (51.2093, 3.2247),
    "Namur": (50.4674, 4.8718), "Leuven": (50.8798, 4.7005),
    "Mons": (50.4542, 3.9513), "Mechelen": (51.0257, 4.4776),
}
city_coords = np.array(list(cities.values()))

def compute_haversine(lat1, lon1, lat2, lon2):
    """
    Compute Haversine distance between two points.
    Input coordinates should be in degrees.
    Returns distance in kilometers.
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Compute Haversine distance
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Convert to kilometers (Earth's radius ≈ 6371 km)
    return 6371 * c

def min_dist_to_cities(df):
    coords = df[["lat", "lon"]].to_numpy()  # shape (N,2)
    cities_np = city_coords  # shape (M,2)

    # Convert both to radians once
    coords_rad = np.radians(coords)
    cities_rad = np.radians(cities_np)

    # Vectorized distance calculation using broadcasting
    dlat = coords_rad[:, None, 0] - cities_rad[None, :, 0]
    dlon = coords_rad[:, None, 1] - cities_rad[None, :, 1]
    a = np.sin(dlat/2)**2 + np.cos(coords_rad[:, None, 0]) * np.cos(cities_rad[None, :, 0]) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    dist_matrix = 6371 * c  # (N, M)

    return np.nanmin(dist_matrix, axis=1)

# Calculate distances for train and test sets
train["dist_to_city_center"] = min_dist_to_cities(train)
test["dist_to_city_center"] = min_dist_to_cities(test)

# Step 3: Distance to nearest train station using BallTree
train_coords_rad = np.radians(train[["lat", "lon"]].to_numpy())
test_coords_rad = np.radians(test[["lat", "lon"]].to_numpy())

# Get distances to the 3 nearest stations for both train and test data
# Note: distances returned by query are in radians, convert to km
train_distances, train_indices = tree.query(train_coords_rad, k=3)
test_distances, test_indices = tree.query(test_coords_rad, k=3)

# Convert distances from radians to kilometers (Earth's radius ≈ 6371 km)
EARTH_RADIUS_KM = 6371.0
train_distances_km = train_distances * EARTH_RADIUS_KM
test_distances_km = test_distances * EARTH_RADIUS_KM

# Add distance to first, second, and third nearest stations
train["dist_to_train_station"] = train_distances_km[:, 0]
train["dist_to_second_station"] = train_distances_km[:, 1]
train["dist_to_third_station"] = train_distances_km[:, 2]
train["avg_3_station_dist"] = np.mean(train_distances_km, axis=1)

test["dist_to_train_station"] = test_distances_km[:, 0]
test["dist_to_second_station"] = test_distances_km[:, 1]
test["dist_to_third_station"] = test_distances_km[:, 2]
test["avg_3_station_dist"] = np.mean(test_distances_km, axis=1)

# Add station density features - count stations within certain radii
for radius_km in [1, 3, 5]:
    # Convert radius from km to radians for query_radius
    radius_rad = radius_km / EARTH_RADIUS_KM
    
    # Count points within radius for each location
    train_count = tree.query_radius(train_coords_rad, r=radius_rad, count_only=True)
    test_count = tree.query_radius(test_coords_rad, r=radius_rad, count_only=True)
    
    train[f"stations_within_{radius_km}km"] = train_count
    test[f"stations_within_{radius_km}km"] = test_count

print("Added enhanced train station features: distances to multiple stations and station density")

# Step 4: KMeans clustering
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(train[["lat", "lon"]])
train["location_cluster"] = kmeans.predict(train[["lat", "lon"]])
test["location_cluster"] = kmeans.predict(test[["lat", "lon"]])

# Feature transformations for numeric features
# Areas and financial values are often better represented in log scale
# Find columns that exist in both train and test sets that should be transformed
common_cols = [col for col in numeric_cols if col in train.columns and col in test.columns]
area_cols = [col for col in common_cols if 'area' in col.lower()]
price_cols = [col for col in common_cols if 'price' in col.lower() and col != 'price']

# Transform area columns
for col in area_cols:
    if train[col].min() > 0 and test[col].min() > 0:
        train[f'{col}_log'] = np.log1p(train[col])
        test[f'{col}_log'] = np.log1p(test[col])
        print(f"Log-transformed {col} in both datasets")

# Transform price-related columns that exist in both datasets
for col in price_cols:
    if train[col].min() > 0 and test[col].min() > 0:
        train[f'{col}_log'] = np.log1p(train[col])
        test[f'{col}_log'] = np.log1p(test[col])
        print(f"Log-transformed {col} in both datasets")

# Ratio features 
if 'rooms' in train.columns and 'area' in train.columns:
    train['area_per_room'] = train['area'] / train['rooms'].replace(0, 1)
    test['area_per_room'] = test['area'] / test['rooms'].replace(0, 1)

# --- Prepare data for modeling ---
print("\nPreparing data for modeling...")

# Remove unnecessary columns from both datasets if they exist
def safe_drop_columns(df, columns_to_drop):
    """Safely drop columns that exist in the DataFrame"""
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    if existing_columns:
        return df.drop(columns=existing_columns)
    return df

# Define columns to potentially drop
potential_drops = ["id", "price"]

# Safely drop columns
X = safe_drop_columns(train, potential_drops)
X_test = safe_drop_columns(test, ["id"])

# Convert all object columns to category for LightGBM
print("Converting object columns to category...")
for df in [X, X_test]:
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')

print(f"Final feature set shape - Train: {X.shape}, Test: {X_test.shape}")
print(f"Features: {X.columns.tolist()}")

# Set up cross-validation
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Add early stopping callback for LightGBM
early_stopping_callback = lgb.early_stopping(
    stopping_rounds=50,
    verbose=True
)

# Add model checkpoint directory
checkpoint_dir = "model_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Set up AutoML settings
point_prediction_settings = {
    "task": "regression",
    "time_budget": 300,  # 5 minutes for point prediction
    "metric": "mae",
    "estimator_list": ["lgbm", "rf", "xgboost", "extra_tree", "catboost"],
    "n_jobs": -1,
    "verbose": 1,
    "seed": 42,
    "n_splits": n_folds,  # Use our custom CV folds
    "eval_method": "cv"
}

# Settings for quantile regression models
quantile_settings = {
    "task": "regression",
    "time_budget": 120,  # 2 minutes for each quantile model
    "metric": "mae",  # We still use MAE for quantile regression
    "estimator_list": ["lgbm"],  # Only LightGBM supports quantile regression directly
    "n_jobs": -1,
    "verbose": 1,
    "seed": 42,
    "n_splits": n_folds,
    "eval_method": "cv"
}

# Cross-validation predictions
cv_preds_lower = np.zeros(len(X))
cv_preds_upper = np.zeros(len(X))
cv_preds_point = np.zeros(len(X))
test_preds_lower = np.zeros((n_folds, len(X_test)))
test_preds_upper = np.zeros((n_folds, len(X_test)))
test_preds_point = np.zeros((n_folds, len(X_test)))

# Store fold metrics
fold_metrics = []
fitted_automl_models = [] # New list to store fitted automl objects

print(f"Starting {n_folds}-fold cross-validation at {datetime.now().strftime('%H:%M:%S')}")

def create_study_with_pruning(direction='minimize', study_name=None):
    """
    Create an Optuna study with pruning and improved sampling.
    """
    pruner = MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=20,
        interval_steps=10
    )
    
    sampler = TPESampler(
        n_startup_trials=10,
        n_ei_candidates=24,
        seed=42
    )
    
    return optuna.create_study(
        study_name=study_name,
        direction=direction,
        pruner=pruner,
        sampler=sampler
    )

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold+1}/{n_folds}")
    
    # Use a fixed time budget for each fold
    current_time_budget = 300
    print(f"Using time budget of {current_time_budget} seconds for this fold")
    
    # Split data for this fold
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    # Create studies with pruning for lower and upper bounds
    lower_study = create_study_with_pruning(
        direction='minimize',
        study_name=f'lower_bound_fold_{fold}'
    )
    upper_study = create_study_with_pruning(
        direction='minimize',
        study_name=f'upper_bound_fold_{fold}'
    )
    
    # Update the objective functions to include pruning
    def create_lower_objective(X_train, y_train, X_val, y_val):
        def objective(trial):
            # Define hyperparameters to search
            params = {
                'objective': 'quantile',
                'alpha': 0.1,  # 10th percentile
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42
            }
            
            # Create pruning callback with 'quantile' metric instead of 'mae'
            pruning_callback = LightGBMPruningCallback(trial, 'quantile')
            
            # Train model with pruning
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[pruning_callback]
            )
            
            # Get best score
            best_score = model.best_score_['valid_0']['quantile']
            return best_score
            
        return objective

    # Train lower bound model with Optuna optimization
    print("Training lower bound model with Optuna...")

    lower_objective_func = create_lower_objective(X_train_fold, y_train_fold, X_val_fold, y_val_fold)

    # Run Optuna with a time limit
    try:
        lower_study.optimize(lower_objective_func, n_trials=50, timeout=current_time_budget)
        print(f"Best lower bound params: {lower_study.best_params}")
        print(f"Best lower bound value: {lower_study.best_value:.4f}")
        
        # Train the final model with best parameters
        best_params = lower_study.best_params.copy()
        best_params['objective'] = 'quantile'
        best_params['alpha'] = 0.1
        best_params['random_state'] = 42
        
        best_lower_model = lgb.LGBMRegressor(**best_params)
        best_lower_model.fit(X_train_fold, y_train_fold)
        
    except Exception as e:
        print(f"Error in Optuna optimization for lower bound: {str(e)}")
        # No fallback model

    # Use the best model
    lower_preds = best_lower_model.predict(X_val_fold)
    cv_preds_lower[val_idx] = lower_preds
    test_preds_lower[fold] = best_lower_model.predict(X_test)

    # Train upper bound model with Optuna optimization
    print("Training upper bound model with Optuna...")

    def create_upper_objective(X_train, y_train, X_val, y_val):
        """Create an objective function for the upper bound model (90th percentile)"""
        def objective(trial):
            # Define hyperparameters to search
            params = {
                'objective': 'quantile',
                'alpha': 0.9,  # 90th percentile
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42
            }
            
            # Train model
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            preds = model.predict(X_val)
            
            # Custom metric for upper bound: penalize predictions that are too low
            # We want predictions to generally be above the true value (90th percentile)
            mae = np.mean(np.abs(preds - y_val) + 3.0 * np.maximum(0, y_val - preds))
            return mae
        
        return objective

    # Run Optuna with a time limit
    upper_objective_func = create_upper_objective(X_train_fold, y_train_fold, X_val_fold, y_val_fold)

    # Run for at most 5 minutes
    try:
        upper_study.optimize(upper_objective_func, n_trials=50, timeout=current_time_budget)
        print(f"Best upper bound params: {upper_study.best_params}")
        print(f"Best upper bound value: {upper_study.best_value:.4f}")
        
        # Train the final model with best parameters
        best_params = upper_study.best_params.copy()
        best_params['objective'] = 'quantile'
        best_params['alpha'] = 0.9
        best_params['random_state'] = 42
        
        best_upper_model = lgb.LGBMRegressor(**best_params)
        best_upper_model.fit(X_train_fold, y_train_fold)
        
    except Exception as e:
        print(f"Error in Optuna optimization for upper bound: {str(e)}")
        # No fallback model

    # Use the best model
    upper_preds = best_upper_model.predict(X_val_fold)
    cv_preds_upper[val_idx] = upper_preds
    test_preds_upper[fold] = best_upper_model.predict(X_test)
    
    # Train point prediction using FLAML AutoML
    print(f"Training point prediction model with AutoML (Fold {fold+1})...")
    automl = AutoML()
    automl.fit(X_train=X_train_fold, y_train=y_train_fold, **point_prediction_settings)
    fitted_automl_models.append(automl) # Store the fitted automl object
    
    # Store validation predictions
    cv_preds_point[val_idx] = automl.predict(X_val_fold)
    
    # Store test predictions for this fold
    test_preds_point[fold] = automl.predict(X_test)
    
    # Calculate fold metrics
    fold_mae = np.mean(np.abs(cv_preds_point[val_idx] - y_val_fold))
    fold_coverage = np.mean((y_val_fold >= cv_preds_lower[val_idx]) & (y_val_fold <= cv_preds_upper[val_idx]))
    interval_width = np.mean(cv_preds_upper[val_idx] - cv_preds_lower[val_idx])
    
    fold_metrics.append({
        'fold': fold + 1,
        'mae': fold_mae,
        'coverage': fold_coverage,
        'interval_width': interval_width,
        'best_model': automl.best_estimator
    })
    
    print(f"Fold {fold+1} metrics - MAE: {fold_mae:.2f}, Coverage: {fold_coverage:.2%}, Width: {interval_width:.2f}")
    print(f"Best model: {automl.best_estimator}, Best config: {automl.best_config}")

# Replace simple median averaging with weighted ensemble
print("Creating median ensemble predictions...")
test_pred_lower = np.median(test_preds_lower, axis=0)
test_pred_upper = np.median(test_preds_upper, axis=0)
test_pred_point = np.median(test_preds_point, axis=0)

# Print overall cross-validation metrics
cv_mae = np.mean(np.abs(cv_preds_point - y))
cv_coverage = np.mean((y >= cv_preds_lower) & (y <= cv_preds_upper))
cv_interval_width = np.mean(cv_preds_upper - cv_preds_lower)

print("\nOverall CV metrics:")
print(f"Mean Absolute Error: {cv_mae:.2f}")
print(f"Prediction interval coverage: {cv_coverage:.2%}")
print(f"Mean interval width: {cv_interval_width:.2f}")

# Analyze fold metrics
fold_df = pd.DataFrame(fold_metrics)
print("\nFold metrics summary:")
print(fold_df.describe())

# Visualize CV predictions vs actual
try:
    plt.figure(figsize=(10, 6))
    plt.scatter(y, cv_preds_point, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual price')
    plt.ylabel('Predicted price')
    plt.title('Cross-validation predictions vs actual')
    plt.savefig('cv_predictions.png')
    print("Saved prediction plot to cv_predictions.png")
except Exception as e:
    print(f"Error creating plot: {str(e)}")

# Create submission using TEST predictions
submission = pd.DataFrame({
    "ID": test_ids,
    "LOWER": test_pred_lower,
    "UPPER": test_pred_upper,
    "PRED": test_pred_point
})

# Ensure bounds are properly ordered
submission["LOWER"] = submission[["LOWER", "UPPER"]].min(axis=1)
submission["UPPER"] = submission[["LOWER", "UPPER"]].max(axis=1)
submission["PRED"] = submission[["PRED", "LOWER"]].max(axis=1)
submission["PRED"] = submission[["PRED", "UPPER"]].min(axis=1)

# Export CSV for submission
submission.to_csv("automl_submission_4.csv", index=False)
print("Submission file created successfully.")

# Feature importance analysis
def comprehensive_feature_analysis(model, X_train, y_train, X_val, y_val, feature_names):
    """
    Perform comprehensive feature importance analysis using multiple methods.
    Handles different model types (LightGBM, XGBoost, RandomForest, etc.)
    """
    results = {}
    
    # 1. Native feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        results['native_importance'] = pd.Series(
            model.feature_importances_,
            index=feature_names
        ).sort_values(ascending=False)
    elif hasattr(model, 'coef_'):  # For linear models
        results['native_importance'] = pd.Series(
            np.abs(model.coef_),
            index=feature_names
        ).sort_values(ascending=False)
    
    # 2. Permutation importance (works with any model)
    try:
        perm_importance = permutation_importance(
            model, X_val, y_val,
            n_repeats=5,
            random_state=42,
            n_jobs=-1
        )
        results['permutation_importance'] = pd.Series(
            perm_importance.importances_mean,
            index=feature_names
        ).sort_values(ascending=False)
    except Exception as e:
        print(f"Error calculating permutation importance: {str(e)}")
    
    # 3. SHAP values (try different explainers based on model type)
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'estimator'):  # For FLAML AutoML
            base_model = model.model.estimator
            model_type = str(type(base_model).__name__).lower()
            if model_type in ['lgbmregressor', 'lightgbm', 'xgbregressor', 'xgboost']:
                explainer = shap.TreeExplainer(base_model)
                shap_values = explainer.shap_values(X_val.values)
            else:
                # For non-tree models, use KernelExplainer
                explainer = shap.KernelExplainer(base_model.predict, shap.sample(X_val, 100))
                shap_values = explainer.shap_values(X_val.values)
        else:
            # For direct model objects (not AutoML)
            model_type = str(type(model).__name__).lower()
            if model_type in ['lgbmregressor', 'lightgbm', 'xgbregressor', 'xgboost']:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_val.values)
            else:
                explainer = shap.KernelExplainer(model.predict, shap.sample(X_val, 100))
                shap_values = explainer.shap_values(X_val.values)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        results['shap_values'] = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=feature_names
        ).sort_values(ascending=False)
    except Exception as e:
        print(f"Error calculating SHAP values: {str(e)}")
        print("Traceback:")
        import traceback
        traceback.print_exc()

    # Also add native feature importance using estimator directly if available
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'estimator'):
            if hasattr(model.model.estimator, 'feature_importances_'):
                results['native_importance'] = pd.Series(
                    model.model.estimator.feature_importances_,
                    index=feature_names
                ).sort_values(ascending=False)
    except Exception as e:
        print(f"Error getting native feature importance from estimator: {str(e)}")
    
    return results

# Analyze feature importance for each fold
print("\nPerforming feature importance analysis...")
all_importance_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    if fold < len(fold_metrics):
        best_model_name = fold_metrics[fold]['best_model']
        print(f"\nAnalyzing feature importance for fold {fold+1} ({best_model_name})")
        
        # Retrieve the fitted automl model for this fold
        current_automl_model = fitted_automl_models[fold]
        
        X_fold = X.iloc[train_idx]
        y_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        try:
            # Get the actual model object from the stored AutoML instance
            # The actual estimator is usually in automl.model.estimator
            model_to_analyze = None
            if hasattr(current_automl_model, 'model') and hasattr(current_automl_model.model, 'estimator'):
                model_to_analyze = current_automl_model.model.estimator
            else: # Fallback for older FLAML versions or different structures if needed
                model_to_analyze = current_automl_model.model 

            if model_to_analyze is not None:
                # Get comprehensive feature importance
                importance_results = comprehensive_feature_analysis(
                    model_to_analyze, X_fold, y_fold, X_val_fold, y_val_fold, X.columns
                )
                all_importance_results.append(importance_results)
            else:
                print(f"Could not retrieve model for feature analysis from AutoML object for fold {fold+1}")
                
        except Exception as e:
            print(f"Error in feature analysis for fold {fold+1}: {str(e)}")

# Aggregate results across folds
if all_importance_results:
    print("\nAggregating feature importance across folds...")
    
    # Initialize dictionaries for each importance type
    aggregated_importance = {
        'native': pd.DataFrame(),
        'shap': pd.DataFrame(),
        'permutation': pd.DataFrame()
    }
    
    # Collect results from each fold
    for fold_result in all_importance_results:
        if 'native_importance' in fold_result:
            aggregated_importance['native'] = pd.concat(
                [aggregated_importance['native'], 
                 fold_result['native_importance']], axis=1)
        if 'shap_values' in fold_result:
            aggregated_importance['shap'] = pd.concat(
                [aggregated_importance['shap'], 
                 fold_result['shap_values']], axis=1)
        if 'permutation_importance' in fold_result:
            aggregated_importance['permutation'] = pd.concat(
                [aggregated_importance['permutation'], 
                 fold_result['permutation_importance']], axis=1)
    
    # Calculate mean importance across folds
    final_importance = {}
    for imp_type, imp_df in aggregated_importance.items():
        if not imp_df.empty:
            final_importance[imp_type] = imp_df.mean(axis=1).sort_values(ascending=False)
    
    # Plot results
    for imp_type, imp_series in final_importance.items():
        plt.figure(figsize=(12, 8))
        sns.barplot(x=imp_series.head(20).values, y=imp_series.head(20).index)
        plt.title(f'Top 20 Features by {imp_type.title()} Importance')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{imp_type}.png')
        print(f"\nTop 20 features by {imp_type} importance:")
        print(imp_series.head(20))
        
    print("\nFeature importance plots have been saved.")
else:
    print("No feature importances could be extracted")