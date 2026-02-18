import threading
import numpy as np
import pandas as pd
import logging
import re
import joblib
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.utils import resample

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineeringAgent(BaseEstimator, TransformerMixin):
    """
    FeatureEngineeringAgent v10.1.7 (Masterpiece Edition)

    ✔ %100 Özellik Koruma: NLP, Cyclic Date, High-Cardinality, Memory Management.
    ✔ Import Fix: Dosya ismi 'agent.py' olarak standartlaştırıldı.
    ✔ Sektörel Standart: Üretim ortamı için optimize edilmiş, thread-safe mimari.
    """

    __version__ = "10.1.7"

    def __init__(
            self,
            target_column=None,
            do_feature_selection=True,
            feature_selection_k=25,
            problem_type="classification",
            do_scaling=True,
            max_ohe_unique=50,
            stateless_transform=False,
            handle_outliers=True,
            time_series_features=True,
            auto_retrain_drift=False,
            fit_sampling_limit=500000
    ):
        self.target_column = target_column
        self.do_feature_selection = do_feature_selection
        self.feature_selection_k = feature_selection_k
        self.problem_type = problem_type
        self.do_scaling = do_scaling
        self.max_ohe_unique = max_ohe_unique
        self.stateless_transform = stateless_transform
        self.handle_outliers = handle_outliers
        self.time_series_features = time_series_features
        self.auto_retrain_drift = auto_retrain_drift
        self.fit_sampling_limit = fit_sampling_limit

        self._lock = threading.Lock()
        self._is_fitted = False

        self.state = {
            "dropped_columns": [],
            "datetime_columns": [],
            "numeric_cols": [],
            "categorical_cols": [],
            "text_cols": [],
            "high_cardinality_map": {},
            "imputation_values": {},
            "onehot_encoders": {},
            "onehot_feature_names": {},
            "scaler": None,
            "scaler_type": None,
            "scaler_columns": [],
            "outlier_bounds": {},
            "raw_train_stats": {},
            "numeric_shift_proxy": {},
            "final_columns": []
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        if "_lock" in state: del state["_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def _downcast_dtypes(self, df):
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer' if df[col].dtype == 'int64' else 'float')
        return df

    def _is_date_strict(self, series):
        if series.dtype == "datetime64[ns]": return True
        if series.dtype != "object": return False
        sample = series.dropna().head(100).astype(str)
        if sample.empty: return False
        date_pattern = re.compile(r'^\d{4}[-/]\d{2}[-/]\d{2}')
        if sample.apply(lambda x: bool(date_pattern.match(x))).mean() < 0.7: return False
        try:
            return pd.to_datetime(sample, errors="coerce").notna().mean() >= 0.7
        except:
            return False

    def _select_best_scaler(self, df, cols):
        if not cols: return None, None
        outlier_counts = []
        for col in cols:
            if col in df.columns:
                m, s = df[col].mean(), df[col].std() + 1e-9
                outlier_counts.append(((df[col] - m).abs() > 3 * s).mean())
        if outlier_counts and np.mean(outlier_counts) > 0.05:
            return RobustScaler(), "robust"
        return StandardScaler(), "standard"

    def _extract_light_nlp(self, df, col):
        df[f"{col}_len"] = df[col].astype(str).apply(len).astype(np.float32)
        df[f"{col}_words"] = df[col].astype(str).apply(lambda x: len(x.split())).astype(np.float32)
        return [f"{col}_len", f"{col}_words"]

    def _cyclic_encode(self, df, col, max_val):
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_val).astype(np.float32)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_val).astype(np.float32)
        return [f"{col}_sin", f"{col}_cos"]

    def fit(self, df: pd.DataFrame, y=None):
        logger.info(f"Fitting Masterpiece Agent v{self.__version__}...")
        df = self._downcast_dtypes(df.copy())
        df_fit = df.sample(n=min(len(df), self.fit_sampling_limit), random_state=42).copy()
        y_fit = y.loc[df_fit.index] if y is not None and isinstance(y, (pd.Series, pd.DataFrame)) else y

        with self._lock:
            # 1. Analiz ve Kategorizasyon
            date_cols = [c for c in df_fit.columns if c != self.target_column and self._is_date_strict(df_fit[c])]
            to_drop, text_cols, num_cols, cat_cols = [], [], [], []

            for col in df_fit.columns:
                if col == self.target_column or col in date_cols: continue
                nunique = df_fit[col].nunique()

                if nunique == len(df_fit) and any(k in str(col).lower() for k in ["id", "sno", "index", "uid"]):
                    to_drop.append(col);
                    continue
                if nunique <= 1: to_drop.append(col); continue

                if np.issubdtype(df_fit[col].dtype, np.number):
                    num_cols.append(col)
                else:
                    avg_len = df_fit[col].astype(str).apply(len).mean()
                    if avg_len > 20:
                        text_cols.append(col)
                    else:
                        cat_cols.append(col)

            self.state.update({"dropped_columns": to_drop, "numeric_cols": num_cols, "categorical_cols": cat_cols,
                               "datetime_columns": date_cols, "text_cols": text_cols})

            # 2. Imputation Değerleri
            for col in num_cols:
                self.state["imputation_values"][col] = df_fit[col].median()
                self.state["raw_train_stats"][col] = {"mean": float(df_fit[col].mean()),
                                                      "std": float(df_fit[col].std() + 1e-9)}
            for col in cat_cols + text_cols:
                self.state["imputation_values"][col] = df_fit[col].mode()[0] if not df_fit[
                    col].mode().empty else "Missing"

            # 3. Encoding (OHE & High-Cardinality)
            ohe_names = []
            for col in cat_cols:
                if df_fit[col].nunique() <= self.max_ohe_unique:
                    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32).fit(
                        df_fit[[col]].astype(str).fillna(self.state["imputation_values"][col]))
                    self.state["onehot_encoders"][col] = ohe
                    names = ohe.get_feature_names_out([col]).tolist()
                    self.state["onehot_feature_names"][col] = names
                    ohe_names.extend(names)
                else:
                    self.state["high_cardinality_map"][col] = df_fit[col].value_counts(normalize=True).to_dict()

            # 4. Özellik Genişletme (TS & NLP)
            temp_df = df_fit[num_cols].fillna(self.state["imputation_values"]).astype(np.float32)
            expanded = []
            if self.time_series_features:
                for col in date_cols:
                    dt = pd.to_datetime(df_fit[col], errors="coerce").ffill().bfill()
                    temp_df[f"{col}_month"] = dt.dt.month
                    temp_df[f"{col}_dow"] = dt.dt.dayofweek
                    expanded.extend(self._cyclic_encode(temp_df, f"{col}_month", 12))
                    expanded.extend(self._cyclic_encode(temp_df, f"{col}_dow", 7))

            for col in text_cols:
                # Add text column to temp_df for extraction
                temp_df[col] = df_fit[col].fillna(self.state["imputation_values"][col])
                expanded.extend(self._extract_light_nlp(temp_df, col))
                temp_df = temp_df.drop(columns=[col])  # Drop raw text after extraction

            full_numeric_set = temp_df.columns.tolist()

            # 5. Ölçeklendirme
            if self.do_scaling and full_numeric_set:
                scaler, s_type = self._select_best_scaler(temp_df, full_numeric_set)
                self.state["scaler"] = scaler.fit(temp_df[full_numeric_set])
                self.state["scaler_type"], self.state["scaler_columns"] = s_type, full_numeric_set

            # 6. Özellik Seçimi
            selected = full_numeric_set
            if self.do_feature_selection and y_fit is not None:
                try:
                    k = min(self.feature_selection_k, temp_df.shape[1])
                    sel = SelectKBest(
                        mutual_info_regression if self.problem_type == "regression" else mutual_info_classif, k=k).fit(
                        temp_df, y_fit)
                    selected = temp_df.columns[sel.get_support()].tolist()
                except:
                    pass

            self.state["selected_features"] = selected
            self.state["final_columns"] = selected + list(self.state["high_cardinality_map"].keys()) + ohe_names
            self._is_fitted = True
            return self

    def transform(self, df: pd.DataFrame, detect_drift=False):
        if not self._is_fitted: raise RuntimeError("Agent not fitted.")
        if len(df) > 100000:
            return pd.concat(
                [self._transform_batch(c, detect_drift) for c in np.array_split(df, max(1, len(df) // 50000))])
        return self._transform_batch(df, detect_drift)

    def _transform_batch(self, df, detect_drift):
        df = self._downcast_dtypes(df.copy())
        current_drift = {}

        for col in self.state["numeric_cols"]:
            if col in df.columns:
                df[col] = df[col].fillna(self.state["imputation_values"][col])
                if detect_drift and not self.stateless_transform:
                    st = self.state["raw_train_stats"].get(col)
                    if st and st["std"] > 0:
                        z = abs(df[col].mean() - st["mean"]) / st["std"]
                        if z > 3.0: current_drift[col] = float(round(z, 2))

        if detect_drift:
            with self._lock: self.state["numeric_shift_proxy"] = current_drift

        df = df.drop(columns=self.state["dropped_columns"], errors="ignore")

        for col in self.state["datetime_columns"]:
            if col in df.columns:
                dt = pd.to_datetime(df[col], errors="coerce").ffill().bfill()
                df[f"{col}_month"] = dt.dt.month
                df[f"{col}_dow"] = dt.dt.dayofweek
                self._cyclic_encode(df, f"{col}_month", 12)
                self._cyclic_encode(df, f"{col}_dow", 7)

        for col in self.state["text_cols"]:
            if col in df.columns: self._extract_light_nlp(df, col)

        for col, fmap in self.state["high_cardinality_map"].items():
            if col in df.columns: df[col] = df[col].map(fmap).fillna(0)

        parts = []
        for col, ohe in self.state["onehot_encoders"].items():
            if col in df.columns:
                val = df[[col]].astype(str).fillna(self.state["imputation_values"][col])
                parts.append(
                    pd.DataFrame(ohe.transform(val), columns=self.state["onehot_feature_names"][col], index=df.index))
        if parts: df = pd.concat([df] + parts, axis=1)

        if self.state["scaler"]:
            cls = [c for c in self.state["scaler_columns"] if c in df.columns]
            if cls: df[cls] = self.state["scaler"].transform(df[cls].fillna(0))

        for m in self.state["final_columns"]:
            if m not in df.columns: df[m] = 0.0

        return df[self.state["final_columns"]]

    def get_drift_report(self):
        with self._lock: return self.state["numeric_shift_proxy"].copy()
