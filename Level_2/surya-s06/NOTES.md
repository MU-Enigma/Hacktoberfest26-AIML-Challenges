# NOTES – How to Run the Project

## 1. Prerequisites

Make sure you have the following installed:

- Python 3
- VS Code (recommended)
- Python extension for VS Code
- Jupyter extension for VS Code

The project uses three Python libraries:

- NumPy
- Pandas
- Matplotlib

## 2. Open the Project

Open the `Hacktoberfest26-AIML-Challenges` repository folder in VS Code.

Open the VS Code terminal:

**Terminal → New Terminal**

Make sure the terminal is at the repository root:

```powershell
C:\Users\Surya\Documents\Hacktoberfest26-AIML-Challenges
```

## 3. Install Required Libraries

Run:

```powershell
python -m pip install numpy pandas matplotlib
```

To verify the installation:

```powershell
python -c "import numpy, pandas, matplotlib; print('All libraries installed successfully!')"
```

## 4. Run the Logistic Regression Program

From the repository root, execute:

```powershell
python .\Level_2\surya-s06\logistic_regression.py
```

The program will:

1. Load the binary classification dataset.
2. Split it into training and testing sets (80/20).
3. Standardize the input features.
4. Train logistic regression models using three learning rates:
   - 0.001
   - 0.05
   - 0.5
5. Calculate test accuracy and final loss.
6. Check whether each model converged.
7. Display three incorrectly classified examples.
8. Test the effect of removing the `handwriting_neatness` feature.

## 5. View Generated Outputs

After running the program, check the `outputs` folder inside `surya-s06`.

It contains:

- `loss_curves.png` – A graph comparing the loss curves of the three learning rates.
- `misclassified_examples.csv` – Three incorrectly classified test examples.

## 6. Run the Jupyter Notebook

Open:

```text
Level_2/surya-s06/teaching_notebook.ipynb
```

In VS Code:

1. Open the notebook.
2. Select the Python interpreter used to install the libraries.
3. Select the Jupyter kernel.
4. Click **Run All** to execute every cell.

The notebook explains logistic regression, the sigmoid function, binary cross-entropy, gradient descent, and learning-rate comparisons step by step.

Make sure the notebook's working directory is `Level_2/surya-s06` so its relative dataset path resolves correctly.

## 7. Expected Experiment Results

Using the supplied dataset and fixed random seed, the Python script should produce:

| Learning Rate | Test Accuracy | Epochs | Converged |
|---|---:|---:|---|
| 0.001 | 82% | 10000 | No |
| 0.05 | 82% | 2334 | Yes |
| 0.5 | 82% | 294 | Yes |

Small differences in execution time are normal because they depend on the computer.

## 8. Troubleshooting

**Problem: ModuleNotFoundError**

Install the missing libraries:

```powershell
python -m pip install numpy pandas matplotlib
```

**Problem: Dataset file not found**

Make sure you are running the Python script from the repository root and that the `datasets` folder exists.

**Problem: Notebook kernel not found**

Select the correct Python interpreter and install the Jupyter extension in VS Code.

**Problem: Output folder or files are missing**

Run the Python script again. It automatically creates the `outputs` folder and saves the generated files.

## 9. Important Note

The model uses a fixed random seed (`42`) to make the train-test split reproducible.

The experiment is implemented using NumPy, Pandas and Matplotlib without using a prebuilt logistic regression estimator.