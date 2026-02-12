# Classifier Module

Unified classifier infrastructure for prompt mining classification tasks.

## Quick Start

```python
from prompt_mining.classifiers import (
    ClassificationDataset,
    LinearClassifier,
    XGBoostClassifier,
    DANNClassifier,
    lodo_evaluate,
)

# Load data
dataset = ClassificationDataset.from_path("/path/to/ingestion/output")
data = dataset.load(layer=31, space='raw')

# Train and evaluate
clf = LinearClassifier()
results = lodo_evaluate(clf, data.X, data.y, data.datasets, use_cv_threshold=True)
print(results.summary())
```

## Module Structure

| File | Contents |
|------|----------|
| [base_classifier.py](base_classifier.py) | `ClassifierProtocol`, `evaluate_predictions` |
| [dataset.py](dataset.py) | `ClassificationDataset`, `ClassificationData` |
| [sklearn_classifiers.py](sklearn_classifiers.py) | `LinearClassifier`, `XGBoostClassifier` and configs |
| [dann.py](dann.py) | `DANNClassifier`, `DANNConfig`, `DANNModule` |
| [feature_selection.py](feature_selection.py) | `npmi_feature_selection`, `compute_npmi` |
| [lodo.py](lodo.py) | `lodo_evaluate`, `LODOResult` |

## Components

### ClassificationDataset

Loads activation data with labels and dataset IDs.

```python
dataset = ClassificationDataset.from_path("/path/to/ingestion/output")

# Load raw activations
data = dataset.load(layer=31, space='raw')
# data.X: (n_samples, n_features) feature matrix
# data.y: (n_samples,) binary labels
# data.datasets: (n_samples,) dataset IDs
# data.run_ids: (n_samples,) run IDs for traceability

# Get prompt texts
prompts = dataset.get_prompts(data, indices=[0, 1, 2])
```

### LinearClassifier

Sklearn-based linear classifier with normalization.

```python
from prompt_mining.classifiers import LinearClassifier, LinearConfig

clf = LinearClassifier(LinearConfig(
    model='logistic',      # 'logistic' or 'sgd'
    normalize='standard',  # 'l2', 'standard', or 'none'
    C=1.0,
))

clf.fit(X_train, y_train)
scores = clf.predict_scores(X_test)
```

### XGBoostClassifier

Gradient boosted trees for classification.

```python
from prompt_mining.classifiers import XGBoostClassifier, XGBoostConfig

clf = XGBoostClassifier(XGBoostConfig(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    normalize='none',
))

clf.fit(X_train, y_train)
scores = clf.predict_scores(X_test)
```

### DANNClassifier

Domain-adversarial neural network for cross-dataset generalization.

```python
from prompt_mining.classifiers import DANNClassifier, DANNConfig

clf = DANNClassifier(DANNConfig(
    normalize='standard',
    domain_weight=0.5,  # Weight for domain adversarial loss
    max_epochs=40,
))

clf.fit(X_train, y_train, datasets=dataset_ids)
scores = clf.predict_scores(X_test)
```

### lodo_evaluate

Leave-One-Dataset-Out evaluation protocol.

```python
from prompt_mining.classifiers import lodo_evaluate

results = lodo_evaluate(
    classifier,
    data.X,
    data.y,
    data.datasets,
    target_precision=0.95,
    merge_datasets={'gandalf_summarization': 'mosscap'},
    use_cv_threshold=True,  # Use CV for sklearn classifiers
)

print(results.summary())
# Access per-dataset results
print(results.per_dataset['injecagent']['malicious_f1'])
```

### Feature Selection

NPMI-based feature selection for cross-dataset generalization.

```python
from prompt_mining.classifiers import npmi_feature_selection

# Select class-predictive but dataset-agnostic features
mask = npmi_feature_selection(
    X_binary, y, datasets,
    class_threshold=0.1,    # Min |NPMI| with class
    dataset_threshold=0.5,  # Max |NPMI| with any dataset
)

X_selected = X[:, mask]
```

## Usage Notebook

See the `demos/` directory for complete usage examples.

## Backward Compatibility

Old imports from `prompt_mining.analysis` are deprecated but still work:

```python
# Deprecated - shows warning
from prompt_mining.analysis import DANN, train_dann_lodo

# Preferred
from prompt_mining.classifiers import DANNClassifier, lodo_evaluate
```
