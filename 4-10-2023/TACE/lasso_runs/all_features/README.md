# Overview

Run lasso regression on all 112 extracted features
- correlated features **not** removed
- Ran on both features extracted from ground truth and predicted segmentations
- Grid search over alpha in `1.9-2.0`
- To replicate, run `lasso_sel.py`

Experiment setup
- 5-fold cross validation
- Select best average MSE over all alphas
- Select all features corresponding to nonzero weights
- Repeat 100 times, view frequency of selected features

Purpose for repeating CV 100 times
- assess stability of selected features (compare with KL div)
- Identify if multicollinear features are selected over folds
  - Assess need for tie-breaking with KL div

## Results (gt)

Metrics
- Avg MSE: `47.665 ± 15.197`
- Avg alpha: `1.946 ± 0.027`

Top 10 most frequently selected features (ascending)
```
[
    'original_glrlm_ShortRunLowGrayLevelEmphasis',
    'original_shape_LeastAxisLength',
    'original_shape_SurfaceVolumeRatio',
    'original_shape_SurfaceArea',
    'original_shape_Maximum2DDiameterRow',
    'original_glrlm_GrayLevelNonUniformityNormalized',
    'original_shape_Maximum2DDiameterColumn',
    'original_gldm_DependenceEntropy',
    'original_glrlm_RunEntropy',
    'original_glszm_LowGrayLevelZoneEmphasis'
]
```

Results (predicted)
- Avg MSE: `48.852 ± 16.290`
- Avg alpha: `1.949 ± 0.0298`