from __future__ import annotations

from typing import Dict, List

import numpy as np
import shap


def compute_shap_values(
    model_predict_fn,
    background_samples: np.ndarray,
    eval_samples: np.ndarray,
    feature_names: List[str],
) -> Dict[str, np.ndarray]:
    """
    Generic SHAP explanation wrapper.

    model_predict_fn: function that maps np.ndarray -> np.ndarray predictions.
    """
    explainer = shap.KernelExplainer(model_predict_fn, background_samples)
    shap_values = explainer.shap_values(eval_samples, nsamples=100)

    return {
        "shap_values": np.array(shap_values),
        "background": background_samples,
        "eval": eval_samples,
        "feature_names": np.array(feature_names),
    }

