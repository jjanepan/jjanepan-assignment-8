import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist
import os
import uuid

# Directory for storing results
RESULTS_DIR = "./static"

def create_directory(directory):
    """Ensure the result directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def clean_old_files(directory, prefixes=("dataset_", "parameters_")):
    """Remove old files with specified prefixes."""
    for filename in os.listdir(directory):
        if filename.startswith(prefixes):
            os.remove(os.path.join(directory, filename))

def generate_clusters(distance, n_samples=100, cluster_std=0.5):
    """Generate two ellipsoid clusters separated by a given distance."""
    np.random.seed(0)
    cov_matrix = np.array([[cluster_std, cluster_std * 0.8], [cluster_std * 0.8, cluster_std]])

    # Generate first cluster
    X1 = np.random.multivariate_normal([1, 1], cov_matrix, size=n_samples)
    y1 = np.zeros(n_samples)

    # Generate second cluster and apply shift
    X2 = np.random.multivariate_normal([1, 1], cov_matrix, size=n_samples) + distance
    y2 = np.ones(n_samples)

    # Combine clusters
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    return X, y

def fit_logistic_model(X, y):
    """Fit a logistic regression model and return its parameters."""
    model = LogisticRegression()
    model.fit(X, y)
    return model, model.intercept_[0], *model.coef_[0]

def plot_results(shift_distances, results, output_path):
    """Generate plots for logistic regression results."""
    plt.figure(figsize=(18, 15))

    metrics = ["Beta0", "Beta1", "Beta2", "Slope", "Intercept", "Loss", "Margin Width"]
    for i, (metric, values) in enumerate(results.items(), start=1):
        plt.subplot(3, 3, i)
        plt.plot(shift_distances, values, marker="o")
        plt.title(f"Shift Distance vs {metric}")
        plt.xlabel("Shift Distance")
        plt.ylabel(metric)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def do_experiments(start, end, steps):
    """Perform experiments for logistic regression with shifted clusters."""
    create_directory(RESULTS_DIR)
    clean_old_files(RESULTS_DIR)

    # Generate shift distances
    shift_distances = np.linspace(start, end, steps)

    # Initialize result storage
    results = {
        "Beta0": [], "Beta1": [], "Beta2": [],
        "Slope": [], "Intercept": [], "Loss": [], "Margin Width": []
    }

    # Prepare for dataset plotting
    dataset_filename = f"dataset_{uuid.uuid4().hex}.png"
    dataset_path = os.path.join(RESULTS_DIR, dataset_filename)
    plt.figure(figsize=(15, steps * 5))

    for i, distance in enumerate(shift_distances):
        X, y = generate_clusters(distance)
        model, beta0, beta1, beta2 = fit_logistic_model(X, y)

        # Calculate metrics
        slope = -beta1 / beta2
        intercept = -beta0 / beta2
        y_prob = np.clip(model.predict_proba(X)[:, 1], 1e-10, 1 - 1e-10)
        loss = -np.mean(y * np.log(y_prob) + (1 - y) * np.log(1 - y_prob))

        # Store metrics
        results["Beta0"].append(beta0)
        results["Beta1"].append(beta1)
        results["Beta2"].append(beta2)
        results["Slope"].append(slope)
        results["Intercept"].append(intercept)
        results["Loss"].append(loss)

        # Calculate margin width using contour distances
        xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
                             np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200))
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
        contours = plt.contour(xx, yy, Z, levels=[0.5], colors="black")
        margin_width = np.min(cdist(contours.collections[0].get_paths()[0].vertices,
                                    contours.collections[0].get_paths()[0].vertices))
        results["Margin Width"].append(margin_width)

        # Plot dataset and decision boundary
        plt.subplot(steps, 1, i + 1)
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c="blue", label="Class 0")
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c="red", label="Class 1")
        plt.plot(xx[0, :], slope * xx[0, :] + intercept, "k--", label="Decision Boundary")
        plt.title(f"Shift Distance: {distance:.2f}")
        plt.legend()

    plt.tight_layout()
    plt.savefig(dataset_path)
    plt.close()

    # Plot metrics
    parameters_filename = f"parameters_{uuid.uuid4().hex}.png"
    parameters_path = os.path.join(RESULTS_DIR, parameters_filename)
    plot_results(shift_distances, results, parameters_path)

    return os.path.basename(dataset_path), os.path.basename(parameters_path)

if __name__ == "__main__":
    do_experiments(0.25, 2.0, 8)
