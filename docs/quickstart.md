# Quickstart

A minimal end-to-end example: load a bundled dataset, fit a model,
evaluate on a held-out test split, and inspect calibration.

```python
from sklearn.model_selection import train_test_split

from clinikit.datasets import load_pima
from clinikit.metrics import sensitivity, specificity
from clinikit.models import RuleAugmentedClassifier

X, y = load_pima(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RuleAugmentedClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Sensitivity:", sensitivity(y_test, y_pred))
print("Specificity:", specificity(y_test, y_pred))
```

See the [examples folder](https://github.com/clinikit/clinikit/tree/main/examples)
for longer, runnable notebooks.
