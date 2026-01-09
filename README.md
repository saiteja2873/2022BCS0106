## Steps to Reproduce Lab 2: Automated Training and Metric Reporting using GitHub Actions

### 1. Repository Setup

1. Create a GitHub account and a public repository named `2022BCS0106`.
2. Create the following directory structure:

   ```
   2022BCS0106/
   ├── dataset/
   ├── outputs/
   ├── .github/workflows/
   ├── train.py
   ├── requirements.txt
   ```
3. Place the Wine Quality dataset inside the `dataset/` directory.

---

### 2. Training Script Implementation

1. Write a `train.py` script that:

   * Loads the dataset
   * Applies experiment-specific preprocessing and feature selection
   * Trains the selected regression model
   * Computes Mean Squared Error (MSE) and R² score
   * Prints metrics to standard output
   * Saves the trained model to `outputs/`
   * Saves evaluation metrics to `outputs/results.json`
2. Ensure all experiment configurations are hardcoded in `train.py` and not controlled via the workflow file.

---

### 3. Dependency Management

1. Create a `requirements.txt` file listing all required Python packages.
2. Ensure the project can be installed using:

   ```
   pip install -r requirements.txt
   ```

---

### 4. GitHub Actions Workflow

1. Create a workflow file (e.g., `train.yml`) inside `.github/workflows/`.
2. Configure the workflow to:

   * Trigger on push and pull request to the `main` branch
   * Set up a Python environment
   * Install dependencies
   * Run `train.py`
   * Write metrics to the GitHub Actions Job Summary
   * Upload trained model and results JSON as artifacts

---

### 5. Running Experiments (Core Lab Requirement)

1. For each experiment from Lab 1:

   * Modify `train.py` according to the experiment configuration
   * Commit the changes directly to the `main` branch
   * Use a descriptive commit message indicating model, preprocessing, and split
   * Push the commit to GitHub
2. Each push triggers a separate GitHub Actions run representing one experiment.

---

### 6. Verification

For every workflow run:

1. Open the GitHub Actions tab and select the run.
2. Verify:

   * Job Summary displays MSE and R²
   * Artifacts include the trained model and results JSON
3. Download artifacts to confirm correctness.

---

### 7. Documentation and Analysis

1. Capture screenshots of:

   * Job Summaries with metrics
   * Artifact download sections
2. Answer the analysis questions focusing on:

   * Reproducibility via CI
   * Ease of comparison across runs
   * Role of Git commit history
   * Benefits over manual experimentation
   * Limitations of the approach

---

### 8. Final Submission

Submit:

* GitHub repository link
* Screenshots of workflow runs and artifacts
* Written analysis answers

---