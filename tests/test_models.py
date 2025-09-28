from wine_ai.models.multimodal import MultimodalWineModel
from wine_ai.models.classification import WineClassifier

import numpy as np


def test_multimodal_summary():
    model = MultimodalWineModel()
    summary = model.summary()
    assert "text_backbone" in summary


def test_classifier_predict():
    clf = WineClassifier()
    features = np.array([[0.1, 0.2], [0.5, 0.7], [0.9, 0.1]])
    labels = ["red", "white", "red"]
    clf.fit(features, labels)
    preds = clf.predict(features)
    assert len(preds) == 3
