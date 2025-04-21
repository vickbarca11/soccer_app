# preprocessing_utils.py

# def clean_categories(X):
#     return X.applymap(lambda x: x.lower().replace(" ", "_") if isinstance(x, str) else x)

def clean_categories(X):
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].str.lower().str.replace(" ", "_")
    return X