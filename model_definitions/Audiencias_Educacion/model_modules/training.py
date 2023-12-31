from teradataml import *
from aoa import (
    record_training_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

def train(context: ModelContext, **kwargs):
    aoa_create_context()

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]

    # read training dataset from Teradata and convert to pandas
    train_df = DataFrame.from_query(context.dataset_info.sql)
    train_pdf=train_df.to_pandas(all_rows=True)

    # split data into X and y
    X_train=train_pdf[feature_names]
    y_train=train_pdf[target_name]

    print("Starting training...")

    modelo=Pipeline([('scaler', StandardScaler()),
                      ('RandomForest', RandomForestClassifier(n_estimators=context.hyperparams["n_estimators"],
                                                                criterion=context.hyperparams["criterion"],
                                                                max_features=context.hyperparams["max_features"],
                                                                bootstrap=context.hyperparams["bootstrap"],
                                                                max_samples=context.hyperparams["max_samples"],
                                                                oob_score=context.hyperparams["oob_score"])
                    )])

    Morf=modelo.fit(X_train,y_train)
    
    modelo_calibrado = CalibratedClassifierCV(Morf, cv=5, method='isotonic')
    modelo_calibrado = modelo_calibrado.fit(X_train,y_train)
    joblib.dump(modelo_calibrado,f"{context.artifact_output_path}/model.joblib")
    
    print("Saved trained model")

    plt.barh(X_train.columns,Morf.named_steps["RandomForest"].feature_importances_)
    save_plot("feature_importance.png", context=context)
    feature_importance_val = Morf.named_steps["RandomForest"].feature_importances_
    feature_importance = pd.DataFrame(
                            {'predictor': X_train.columns,
                            'importancia': feature_importance_val}
                            )
    
    record_training_stats(
        train_df,
        features=feature_names,
        targets=[target_name],
        categorical=[target_name],
        feature_importance=feature_importance,
        context=context
    )
