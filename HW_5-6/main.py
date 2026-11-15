# main.py
import logging
import warnings
from rdkit import RDLogger

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
logging.getLogger("deepchem").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from rdkit import Chem
import deepchem as dc
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    mutual_info_classif,
    f_classif,
    SelectFromModel,
    RFE,
)
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, cohen_kappa_score, classification_report


# ---------- 1. Загрузка ----------


def read_sdf_robustly(path, label_field="Mutagenicity"):
    mols, y = [], []
    # removeHs=False — важно для корректного расчёта дескрипторов
    for mol in Chem.SDMolSupplier(path, removeHs=False):
        if mol is None:
            print("mol is None")
            continue  # Пропускаем только "битые" молекулы
        try:
            label = int(float(mol.GetProp(label_field)))
            mols.append(mol)
            y.append(label)
        except (KeyError, ValueError):
            continue  # Пропускаем молекулы без нужной метки
    return mols, np.array(y)


X_mols_train, y_train = read_sdf_robustly("./AMES_MODELING_SET.sdf")
X_mols_test, y_test = read_sdf_robustly("./AMES_TEST_SET.sdf")

featurizer = dc.feat.RDKitDescriptors()

X_train = featurizer.featurize(X_mols_train)
X_test = featurizer.featurize(X_mols_test)
y_train = np.asarray(y_train, dtype=int)  # уже есть из read_sdf
y_test = np.asarray(y_test, dtype=int)

# замена inf -> NaN
X_train = np.where(np.isinf(X_train), np.nan, X_train)
X_test = np.where(np.isinf(X_test), np.nan, X_test)

# ---------- 2. Сравнение нормализации ----------

normalizers = {
    "none": "passthrough",
    "standard": StandardScaler(),
    "minmax": MinMaxScaler(),
    "l2": Normalizer(norm="l2"),
    "robust": RobustScaler(quantile_range=(25.0, 75.0)),
}
norm_rows = []
for name, scaler in normalizers.items():
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler),
        ("var", VarianceThreshold(0.0)),
        ("select", SelectKBest(mutual_info_classif, k=50)),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    norm_rows.append([name, f1_score(y_test, pred), cohen_kappa_score(y_test, pred)])

df_norm = pd.DataFrame(norm_rows, columns=["normalization", "f1", "kappa"])
print("normalizations", df_norm)

# ---------- 3. Сравнение отбора признаков ----------

selectors = {
    "all": "passthrough",
    "var_0.01": VarianceThreshold(0.01),
    "mi_k50": SelectKBest(mutual_info_classif, k=50),
    "f_k50": SelectKBest(f_classif, k=50),
    "sfm_rf_median": SelectFromModel(
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        threshold="median",
    ),
}
sel_rows = []
for name, selector in selectors.items():
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "scaler",
            RobustScaler(quantile_range=(25.0, 75.0)),
        ),
        ("selector", selector),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    n_feat = (
        X_train.shape[1]
        if name == "all"
        else pipe.named_steps["selector"].get_support().sum()
    )
    sel_rows.append([
        name,
        n_feat,
        f1_score(y_test, pred),
        cohen_kappa_score(y_test, pred),
    ])

df_sel = pd.DataFrame(sel_rows, columns=["selector", "n_features", "f1", "kappa"])
print("selectors", df_sel)

# ---------- 4. Финальные модели ----------

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "LogReg": LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ),
    "HistGB": HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        random_state=42,
    ),
}
final_rows = []
for mname, model in models.items():
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "scaler",
            RobustScaler(quantile_range=(25.0, 75.0)),
        ),
        ("var", VarianceThreshold(0.01)),
        (
            "select",
            SelectFromModel(
                RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
                threshold="median",
            ),
        ),
        ("clf", model),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    final_rows.append([mname, f1_score(y_test, pred), cohen_kappa_score(y_test, pred)])
    print(
        f"\n{mname}\n{classification_report(y_test, pred, target_names=['non-mutagen', 'mutagen'])}"
    )

print("models", pd.DataFrame(final_rows, columns=["model", "f1", "kappa"]))
