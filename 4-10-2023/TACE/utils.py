import pandas as pd


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

# select features
f1 = [
    'Study ID',
    'original_shape_Elongation',
    'original_shape_Flatness',
    'original_shape_MajorAxisLength',
    'original_shape_MinorAxisLength',
    'original_shape_Sphericity',
    'original_glrlm_GrayLevelNonUniformity',
    'original_glrlm_LowGrayLevelRunEmphasis',
    'original_glszm_SmallAreaLowGrayLevelEmphasis'
]

f2 = [
    'Study ID',
    'original_shape_Flatness',
    'original_shape_LeastAxisLength',
    'original_shape_Maximum2DDiameterColumn',
    'original_shape_Sphericity',
    'original_shape_SurfaceArea',
    'original_glrlm_GrayLevelNonUniformityNormalized',
    'original_glrlm_RunEntropy',
    'original_glszm_ZoneEntropy'
]


def load_features(feature_dir):
    df = pd.read_csv(feature_dir)
    cleaned_df = df.select_dtypes(include=numerics)
    imdir = df["Image"].str.split("/")

    cases = []
    for i in imdir:
        case_num = int(i[-1].split("_")[1])
        cases.append(case_num)

    # insert case ID
    cleaned_df.insert(0, "Study ID", cases)
    return cleaned_df


def load_targets(targ_dir):
    df = pd.read_csv(targ_dir)
    return df


def create_mats(input_df, target_df):

    input_cols = input_df.columns.copy()
    target_cols = target_df.columns.copy()

    df = input_df.merge(target_df, on="Study ID")
    X = df.loc[:, input_cols].to_numpy()[:, 1:]
    y = df.loc[:, target_cols].to_numpy()[:, 1]  # using only overall survival for now

    return X, y