import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from bayes_opt import BayesianOptimization

# Update the file path to the new location
file_path = r"F:\Research\Data science & Urban geometry\Results\ML\UTCI.csv"

# Load data from the new file
data = pd.read_csv(file_path)

feature_columns = ['OR', 'H/W', 'H', 'NT', 'W']
features = data[feature_columns]
target = data['UTCI']

features_with_dummies = pd.get_dummies(features)
features_train, features_test, target_train, target_test = train_test_split(features_with_dummies, target, test_size=0.3, random_state=2)

def dt_classifier(max_depth, min_samples_split, min_samples_leaf):
    clf = DecisionTreeClassifier(max_depth=int(max_depth),
                                 min_samples_split=int(min_samples_split),
                                 min_samples_leaf=int(min_samples_leaf),
                                 random_state=42)
    clf.fit(features_train, target_train)
    return clf.score(features_test, target_test)

# Define parameter bounds
pbounds = {'max_depth': (1, 20),
           'min_samples_split': (2, 20),
           'min_samples_leaf': (1, 10)}

# Perform Bayesian Optimization
optimizer = BayesianOptimization(
    f=dt_classifier,
    pbounds=pbounds,
    verbose=2,
    random_state=0,
)
optimizer.maximize(init_points=5, n_iter=10)

best_params = optimizer.max['params']
print("Best parameters:", best_params)

# Train Decision Tree with the best parameters
best_dt_model = DecisionTreeClassifier(max_depth=int(best_params['max_depth']),
                                       min_samples_split=int(best_params['min_samples_split']),
                                       min_samples_leaf=int(best_params['min_samples_leaf']),
                                       random_state=42)
best_dt_model.fit(features_train, target_train)
y_pred = best_dt_model.predict(features_test)
print(classification_report(target_test, y_pred))
import shap
import matplotlib.pyplot as plt
import os

# Calculate SHAP values
explainer = shap.TreeExplainer(best_dt_model)
shap_values = explainer.shap_values(features_train)

# Define class names
class_names = ['Moderate heat stress', 'Strong heat stress', 'Extreme heat stress']

# Create the directory to save figures if it doesn't exist
save_dir = r"F:\Research\Data science & Urban geometry\Results\ML\DT\UTCI"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Visualize SHAP summary plot
shap.summary_plot(shap_values, features_train, plot_type="bar", class_names=class_names)

# Save the summary plot
save_path_summary = os.path.join(save_dir, "shap_summary_plot.png")
plt.savefig(save_path_summary)
plt.close()  # Close the plot to release resources

# Create SHAP summary plots for each target separately
for i, target in enumerate(class_names):
    # Parameter Importance
    shap.summary_plot(shap_values[i], features_train, show=False, plot_type="bar")
    plt.title(f"Parameter Importance for {target}")
    save_path_param_importance = os.path.join(save_dir, f"Parameter_Importance_{target}.jpg")
    plt.savefig(save_path_param_importance, dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to release resources

    # Parameter Influence
    shap.summary_plot(shap_values[i], features_train, show=False)
    plt.title(f"Parameter Influence for {target}")
    save_path_param_influence = os.path.join(save_dir, f"Parameter_Influence_{target}.jpg")
    plt.savefig(save_path_param_influence, dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to release resources
