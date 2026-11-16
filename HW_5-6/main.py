import logging
import warnings
from rdkit import RDLogger

# Отключение лишних сообщений
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
logging.getLogger("deepchem").setLevel(logging.ERROR)

import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from rdkit import Chem
import deepchem as dc
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    PowerTransformer,
)
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    mutual_info_classif,
    f_classif,
    SelectFromModel,
    RFE,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score, cohen_kappa_score, classification_report


def read_sdf_robustly(path, label_field="Mutagenicity"):
    # Функция для корректного чтения выданных sdf-файлов
    mols, y = [], []
    # removeHs=False — для корректного расчёта дескрипторов
    for mol in Chem.SDMolSupplier(path, removeHs=False):
        if mol is None:
            print("mol is None")
            continue  # Пропускаем "битые" молекулы
        try:
            label = int(float(mol.GetProp(label_field)))
            mols.append(mol)
            y.append(label)
        except (KeyError, ValueError):
            continue  # Пропускаем молекулы без нужной метки
    return mols, np.array(y)


def main():
    # Чтение train и test датасетов
    X_mols_train, y_train = read_sdf_robustly("./AMES_MODELING_SET.sdf")
    X_mols_test, y_test = read_sdf_robustly("./AMES_TEST_SET.sdf")

    # Получение дескрипторов (только RDKitDescriptors)
    featurizer = dc.feat.RDKitDescriptors()
    X_train = featurizer.featurize(X_mols_train)
    X_test = featurizer.featurize(X_mols_test)

    # Приведение метки класса к целочисленному виду
    y_train = np.asarray(y_train, dtype=int)
    y_test = np.asarray(y_test, dtype=int)

    # замена inf -> NaN
    X_train = np.where(np.isinf(X_train), np.nan, X_train)
    X_test = np.where(np.isinf(X_test), np.nan, X_test)

    # Подбор нормализации (на тесте, в жизни так делать нельзя, все гиперпараметры нужно подбирать на валидации)
    normalizers = {
        "none": "passthrough",  # Без нормализации
        "standard": StandardScaler(),  # Центрирование и масштабирование на стандартное отклонение
        "minmax": MinMaxScaler(),  # Приведение к диапазону [0, 1]
        "l2": Normalizer(norm="l2"),  # L2-нормализация каждого образца
        "robust": RobustScaler(
            quantile_range=(25.0, 75.0)
        ),  # Вычитание медианы и деление на межквартильный размах
        "power_yeojohnson": PowerTransformer(
            method="yeo-johnson"
        ),  # Параметрическое нелинейное преобразование, приближающее распределение к нормальному, стабилизирует дисперсию и уменьшает скошенность
    }
    norm_rows = []
    for name, scaler in normalizers.items():
        pipe = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),  # Импутация - замена NaN (на медиану)
            ("scaler", scaler),  # Нормализация
            ("var", VarianceThreshold(0.0)),  # Убираем константные признаки
            (
                "select",
                SelectKBest(mutual_info_classif, k=100),
            ),  # Непараметрический отбор top-k признаков, связанных с целевой переменной
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300, random_state=42, n_jobs=-1
                ),  # Классификатор (случайный лес)
            ),
        ])
        pipe.fit(X_train, y_train)  # Обучение
        pred = pipe.predict(X_test)  # Предсказание
        norm_rows.append([
            name,
            f1_score(y_test, pred),
            cohen_kappa_score(y_test, pred),
        ])

    df_norm = pd.DataFrame(norm_rows, columns=["Нормализация", "F1", "Cohen’s kappa"])
    print("Подбор нормализации:\n", df_norm)

    # Подбор отбора признаков (на тесте, в жизни так делать нельзя, все гиперпараметры нужно подбирать на валидации)
    selectors = {
        "all": "passthrough",  # Без отбора призгаков
        "var_0.05": VarianceThreshold(0.05),  # Убираем почти константные признаки
        "mi_k100": SelectKBest(
            mutual_info_classif, k=100
        ),  # Непараметрический отбор top-k признаков, связанных с целевой переменной
        "f_k100": SelectKBest(
            f_classif, k=100
        ),  # ANOVA отбор top-k признаков, связанных с целевой переменной
        "rfe": RFE(
            estimator=RandomForestClassifier(n_estimators=200, random_state=42),
            n_features_to_select=100,
            step=10,
        ),  # Рекурсивный отбор признаков по важности для модели
        "sfm_rf_median": SelectFromModel(
            RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            threshold="median",
        ),  # Оставляет признаки с важностью выше порога
    }
    sel_rows = []
    for name, selector in selectors.items():
        pipe = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),  # Импутация - замена NaN (на медиану)
            (
                "scaler",
                RobustScaler(
                    quantile_range=(25.0, 75.0)
                ),  # Вычитание медианы и деление на межквартильный размах
            ),
            ("selector", selector),  # Отбор признаков
            (
                "rf",
                RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
            ),  # Классификатор (случайный лес)
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

    df_sel = pd.DataFrame(
        sel_rows, columns=["Отбор признаков", "Кол-во признаков", "F1", "Cohen’s kappa"]
    )
    print("Подбор отбора признаков:\n", df_sel)

    # Сравнение моделей (на тесте)
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),  # Случайный лес с балансировкой классов
        "LogReg": LogisticRegression(
            C=1.0,
            solver="liblinear",
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),  # Логистическая регрессия
        "SVC": SVC(
            probability=True, class_weight="balanced"
        ),  # Support Vector Classification
        "HistGB": HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
            random_state=42,
        ),  # Алгоритм градиентного бустинга, вдохновленный LightGBM
    }
    final_rows = []
    print("Результаты моделей на тесте:")
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
                    RandomForestClassifier(
                        n_estimators=200, random_state=42, n_jobs=-1
                    ),
                    threshold="median",
                ),
            ),
            ("clf", model),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        final_rows.append([
            mname,
            f1_score(y_test, pred),
            cohen_kappa_score(y_test, pred),
        ])
        print(
            f"\n{mname}\n{classification_report(y_test, pred, target_names=['non-mutagen', 'mutagen'])}"
        )

    print(
        "Сравнение моделей:\n",
        pd.DataFrame(final_rows, columns=["Модель", "F1", "Cohen’s kappa"]),
    )

    # Соорудим ансамбль (дешево и сердито)
    ensemble = VotingClassifier(
        estimators=[
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),  # Случайный лес с балансировкой классов
            ),
            (
                "svc",
                SVC(
                    probability=True, class_weight="balanced"
                ),  # Support Vector Classification
            ),
            (
                "hgb",
                HistGradientBoostingClassifier(
                    max_iter=500,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=50,
                    random_state=42,
                ),  # Алгоритм градиентного бустинга, вдохновленный LightGBM
            ),
        ],
        voting="soft",
        n_jobs=-1,
    )
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
        ("clf", ensemble),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print("Ансамбль", f1_score(y_test, pred), cohen_kappa_score(y_test, pred))


if __name__ == "__main__":
    main()
