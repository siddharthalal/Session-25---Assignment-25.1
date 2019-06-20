# Problem Statement:

We will build the random forest model for predicting house prices from boston data set.

### Code to get data:

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn import datasets
    boston = datasets.load_boston()
    features = pd.DataFrame(boston.data, columns=boston.feature_names)
    targets = boston.target
