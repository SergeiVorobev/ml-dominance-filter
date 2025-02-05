# Dominance Filtering with Machine Learning

This project implements a solution for filtering results based on specific dominance rules using a Machine Learning (ML) approach. The program determines which results should remain in a report and which should be filtered out.

---

## **Problem Statement**

Given a set of results, each represented as three non-negative integers:

- `C` (Inference time)
- `P` (Memory usage)
- `J` (Recognition quality)

We need to filter results based on the following dominance criteria:

1. A result `i` dominates result `j` if:
   - `Ci ≤ Cj`
   - `Pi ≤ Pj`
   - `Ji ≥ Jj`

If a result is dominated by another, it is filtered out. The goal is to determine how many results remain after filtering.

---

## **Solution Overview**

This project solves the problem using a **Machine Learning (ML)** approach:

1. **Training Data**: The program uses labeled data where each result is either marked as "remain" (`1`) or "filtered" (`0`).
2. **ML Model**: A `RandomForestClassifier` from the scikit-learn library is trained on this data to classify results.
3. **Prediction**: The model predicts whether new results should remain or be filtered, based on the dominance rules.

---

## **Features**

- Accepts input results as `(C, P, J)` values.
- Outputs the number of non-dominated (remaining) results.
- Uses scikit-learn's Random Forest for classification.
- Includes an example labeled dataset for training and testing.

---

## **Getting Started**

### **Prerequisites**

Make sure you have the following installed:

- **Python 3.7 or higher**: Ensure that you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).
- **pip**: Python’s package installer. It comes with Python by default.

### **Setting up the Virtual Environment**

1. **Create a virtual environment**:

   Navigate to your project directory and run the following command to create a new virtual environment:
   ```bash
   python3 -m venv venv
   
2. **Activate the virtual environment**:
    On macOS/Linux:
    ``bash
    source venv/bin/activate
    
    On Windows:
    ``bash
    venv\Scripts\activate

3. **Install the required dependencies**:
    Once the virtual environment is activated, install the necessary libraries by running:
    ``bash
    pip install -r requirements.txt

    requirements.txt:
    ``bash
    numpy
    scikit-learn

---

## **How to Run**

1. **Clone the repository**:
    ``bash
    git clone https://github.com/SergeiVorobev/ml-dominance-filter.git
    cd ml-dominance-filter

2. **Set up the virtual environment and activate it (as described above)**.

3. **Run the script**:
    ``bash
    python dominance_filter_ml.py

4. **Input the results when prompted. For example**:
    ``bash
    Enter the number of results: 5
    Enter the results in the format: C P J (space-separated)
    300 2 10
    20 1 2
    4 2 10
    200 1 1
    200 1 11

5. **View the output**:
    ``bash
    Accuracy: 1.00
    Predictions (1 = Remain, 0 = Filtered): [1 0 1 0 1]
    Remaining results: 3

---

## **How It Works**

1. **Training**:
    The labeled dataset (example data) is used to train a Random Forest Classifier.
    Each result is represented as (C, P, J) with a label (1 for remain, 0 for filtered).
    
1. **Prediction**:
    The model predicts the label for each result in the input set.
    The number of 1s (remain) is displayed as the output.

---

## **License**

This project is licensed under the MIT License.

---

## **Contact**

For questions or contributions, feel free to open an issue or submit a pull request!
