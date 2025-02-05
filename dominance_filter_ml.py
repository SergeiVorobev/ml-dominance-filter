import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train_and_predict(data):
    """
    Train a machine learning model to classify results as 'remain' or 'filtered'.
    """
    # Separate the input (C, P, J) and the labels (1 = Remain, 0 = Filtered)
    X_train = np.array([x[:3] for x in data])  # Features (C, P, J)
    y_train = np.array([x[3] for x in data])  # Labels (1 = Remain, 0 = Filtered)

    # Train the RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on the training data itself for testing
    y_pred = model.predict(X_train)
    accuracy = np.mean(y_pred == y_train)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Predictions (1 = Remain, 0 = Filtered): {y_pred}")

    # Count the number of remaining results
    print(f"Remaining results: {sum(y_pred)}")


def get_input_data():
    """
    Get input data from the user and return it as a list of tuples.
    """
    n = int(input("Enter the number of results: "))  # Number of results
    data = []
    
    for i in range(n):
        # Read user input for each result (C, P, J)
        result = input(f"Enter result {i + 1} (C P J): ").split()
        C, P, J = map(int, result)
        
        # Manually assign labels for testing
        # Example of manually labeling: alternate between Remain and Filtered
        if i % 2 == 0:  # Just an example: alternate labeling
            label = 1  # Remain
        else:
            label = 0  # Filtered
        
        data.append((C, P, J, label))
    
    return data


def main():
    """
    Reads input from the user, applies ML-based filtering, and outputs the number of remaining results.
    """
    # Get input data from the user
    labeled_data = get_input_data()

    # Train the model and make predictions
    train_and_predict(labeled_data)


if __name__ == "__main__":
    main()
