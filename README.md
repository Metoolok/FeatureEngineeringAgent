live demo 
https://featureengineeringagent-mxfedphhbyycqpdfttqr2p.streamlit.app/

[Ekran kaydı - 2026-02-18 12-51-57.webm](https://github.com/user-attachments/assets/5e6a7ed2-3d81-4aad-8ee8-f3fee6d9a5b2)


# 🛠️ FeatureForge: FeatureEngineeringAgent (v10.1.7)

### *The "Masterpiece Edition" Data Transformation Engine*

**FeatureForge** is a production-grade, thread-safe feature engineering agent designed to transform raw, messy datasets into highly optimized, training-ready feature sets. It bridges the gap between raw data and high-performance Machine Learning models by automating complex feature extraction, scaling, and selection processes.

---

## 🚀 Key Features

### 🧠 Intelligent Feature Extraction

* **NLP Lite Engine:** Automatically detects text columns and extracts meaningful metrics like `string_length` and `word_count`.
* **Cyclic Temporal Encoding:** Converts Date/Time columns into Sine and Cosine transformations, preserving the periodic nature of months and days of the week.
* **High-Cardinality Management:** Identifies columns with excessive unique values and applies frequency mapping to prevent dimensionality explosion.

### 🛡️ Production-Ready Architecture

* **Thread-Safe Design:** Implements `threading.Lock` to ensure safe operation in multi-threaded production environments.
* **State Persistence:** Fully compatible with `joblib` and `pickle` for seamless deployment.
* **Memory Efficiency:** Built-in downcasting for `int64` and `float64` types to minimize RAM footprint during large-scale transformations.

### 📊 Advanced Statistical Processing

* **Automated Scaling:** Dynamically chooses between `StandardScaler` and `RobustScaler` based on the outlier distribution of each column.
* **Feature Selection:** Utilizes `SelectKBest` with Mutual Information (Regression/Classification) to eliminate noise and keep only the most predictive features.
* **Drift Detection:** Real-time monitoring of statistical shifts between training and inference data.

---

## 📦 Installation & Dependencies

Ensure you have the following libraries installed:

```bash
pip install numpy pandas scikit-learn joblib

```

---

## 🛠 Usage Example

```python
import pandas as pd
from agent import FeatureEngineeringAgent

# 1. Initialize the Agent
agent = FeatureEngineeringAgent(
    target_column='Survived',
    problem_type='classification',
    feature_selection_k=15,
    do_scaling=True
)

# 2. Fit and Transform your training data
df_train = pd.read_csv("titanic.csv")
X_train = agent.fit_transform(df_train, y=df_train['Survived'])

# 3. Transform new data (Inference)
df_test = pd.read_csv("test.csv")
X_test = agent.transform(df_test, detect_drift=True)

# 4. Check for Data Drift
drift_report = agent.get_drift_report()
print(f"Detected Drifts: {drift_report}")

```

---

## ⚙️ Configuration Options

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `target_column` | `str` | `None` | The column to be predicted. |
| `do_feature_selection` | `bool` | `True` | Enables SelectKBest filtering. |
| `feature_selection_k` | `int` | `25` | Number of top features to retain. |
| `problem_type` | `str` | `"classification"` | `"classification"` or `"regression"`. |
| `do_scaling` | `bool` | `True` | Automatically scales numeric features. |
| `handle_outliers` | `bool` | `True` | Statistically manages extreme values. |

---

## 🏗 Technical Pipeline Flow

1. **Categorization:** Automatically separates numeric, categorical, datetime, and text columns.
2. **Imputation:** Robust median/mode filling for missing values.
3. **Expansion:** Generation of NLP and Cyclic features.
4. **Encoding:** One-Hot Encoding for low-cardinality and Frequency Encoding for high-cardinality categories.
5. **Refinement:** Scaling and Mutual Information-based feature selection.

---

## 📜 Version History

**v10.1.7 (Masterpiece Edition)**

* Optimized Batch Processing for datasets >100k rows.
* Standardized `agent.py` naming convention for easy imports.
* Improved Date-Strict detection regex.

---

## 👨‍💻 Author

**Metin Mert Turan** *AI Engineering Student* *"Silence the noise, amplify the signal. Transforming raw data into mathematical gold."*
