import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import FeatureExtraction




def plot_feature_importance(model, feature_names, top_n=10):
    """
    Plot feature importance for a given model.

    Parameters:
    - model: Trained model with feature_importances_ attribute.
    - feature_names: List of feature names.
    - top_n: Number of top features to display (default is 10).
    """