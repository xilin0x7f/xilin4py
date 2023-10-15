# Author: 赩林, xilin0x7f@163.com
import os
import copy
import numpy as np
import pandas as pd

def get_strong_feature(cv_evaluator, columns, x, y, save_dir, model_name, endswith="_selector"):
    # get strong feature and data, then save

    # 在使用单层的交叉验证时，使用的是pipeline_fitted, 因此进行copy到pipeline_out_fitted，使其和嵌套交叉验证一致
    if not hasattr(cv_evaluator, "pipeline_out_fitted"):
        cv_evaluator.pipeline_out_fitted = cv_evaluator.pipeline_fitted

    feature_selection_counts = np.zeros(len(columns), dtype=int)
    feature_coef_sum = np.zeros(len(columns))
    for fitted_pipeline in cv_evaluator.pipeline_out_fitted:
        selected_steps = [step for name, step in fitted_pipeline.named_steps.items() if name.endswith(endswith)]
        feature_mask = None
        for idx, step in enumerate(selected_steps):
            if idx == 0:
                feature_mask = copy.deepcopy(step.get_support())
            else:
                feature_mask[feature_mask] = feature_mask[feature_mask] & step.get_support()

        feature_selection_counts += feature_mask.astype(int)

        if hasattr(fitted_pipeline[-1], 'coef_'):
            full_coef = np.zeros(len(feature_mask))
            full_coef[feature_mask] = fitted_pipeline[-1].coef_.flatten()
            feature_coef_sum += full_coef

        if hasattr(fitted_pipeline[-1], 'feature_importances_'):
            full_coef = np.zeros(len(feature_mask))
            full_coef[feature_mask] = fitted_pipeline[-1].feature_importances_.flatten()
            feature_coef_sum += full_coef

    feature_selection_freq = feature_selection_counts / len(cv_evaluator.pipeline_out_fitted)
    feature_coef_mean = feature_coef_sum / len(cv_evaluator.pipeline_out_fitted)
    feature_selection_df = pd.DataFrame({
        "feature": columns[feature_selection_freq > 0],
        "freq": feature_selection_freq[feature_selection_freq > 0],
        "mean_coef": feature_coef_mean[feature_selection_freq > 0]
    })
    feature_selection_df.to_excel(os.path.join(save_dir, f"FeatureSelected_{model_name}.xlsx"), index=False)
    data_strong_feature = x[:, feature_selection_freq == 1]
    data_strong_feature = pd.DataFrame(data_strong_feature, columns=columns[feature_selection_freq == 1])
    data_strong_feature["label"] = y
    data_strong_feature.to_excel(os.path.join(save_dir, f"DataStrongFeature_{model_name}.xlsx"), index=False)
    return feature_selection_df
