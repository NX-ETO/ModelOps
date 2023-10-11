from teradataml import *
from aoa import (
    record_training_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib

def train(context: ModelContext, **kwargs):
    aoa_create_context()

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]

    # read training dataset from Teradata and convert to pandas
    train_df = DataFrame.from_query(context.dataset_info.sql)

    print("Starting training...")

    train_pdf=train_df.to_pandas(all_rows=True)
    X_train=train_pdf[feature_names]
    y_train=train_pdf[target_name]

    modelo=RandomForestClassifier(n_estimators=126,
                               criterion="gini",
                               max_features="sqrt",
                               bootstrap=True,
                               max_samples=2/3,
                               oob_score=True)

    Morf=modelo.fit(X_train,y_train)
    modelo_calibrado = CalibratedClassifierCV(Morf, cv=5, method='isotonic')
    modelo_calibrado = modelo_calibrado.fit(X_train,y_train)
    joblib.dump(modelo_calibrado,f"{context.artifact_output_path}/model.joblib")
    
    print("Saved trained model")

    record_training_stats(
        train_df,
        features=feature_names,
        targets=[target_name],
        categorical=[target_name],
        context=context
    )
