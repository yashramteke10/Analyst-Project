from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor

def train_random_forest(
    X_train,
    y_train,
    save_path="models_store/random_forest_model.pkl",
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=1,
):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    return model

def predict_random_forest(model, X):
    return model.predict(X)

def load_random_forest(path="models_store/random_forest_model.pkl"):
    return joblib.load(path)
