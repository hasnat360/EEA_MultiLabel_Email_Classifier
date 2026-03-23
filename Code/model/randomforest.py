import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from model.base import BaseModel
from Config import Config


class RandomForest(BaseModel):
    def __init__(self, model_name: str, embeddings: np.ndarray, y: np.ndarray) -> None:
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.predictions = None
        self.mdl = RandomForestClassifier(
            n_estimators=Config.N_ESTIMATORS,
            random_state=Config.SEED,
            class_weight="balanced_subsample",
        )
        self.data_transform()

    def train(self, data) -> None:
        self.mdl.fit(data.X_train, data.y_train)

    def predict(self, X_test: np.ndarray) -> None:
        self.predictions = self.mdl.predict(X_test)

    def print_results(self, data) -> None:
        acc = accuracy_score(data.y_test, self.predictions)
        print(f"  Accuracy: {acc:.4f}")
        print(classification_report(data.y_test, self.predictions, zero_division=0))
        print("  Confusion Matrix (rows=true, cols=predicted):")
        print(confusion_matrix(data.y_test, self.predictions))

    def data_transform(self) -> None:
        pass
