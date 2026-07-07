
# Panel Regression Results

This folder contains a Python script that estimates the model:

`Cig_it = α + βT_it + γTCI_it + δGDP_it + μ_i + λ_t + ε_it`

where `μ_i` are country fixed effects and `λ_t` are year fixed effects.

## Files

- `Data17.xlsx` — data file
- `panel_regression_report.py` — Python code
- `index.html` — generated HTML report for presentation/GitHub Pages
- `requirements.txt` — packages needed to run the code

## How to run locally

```bash
pip install -r requirements.txt
python panel_regression_report.py
```

The script creates `index.html`. Open it in your browser.

## How to use in presentation

1. Publish this folder with GitHub Pages.
2. Copy the GitHub Pages URL.
3. In PowerPoint, select text like **Full Python Results**.
4. Insert a hyperlink to the GitHub Pages URL.
5. Export the PowerPoint to PDF and test the link.

## Important note

The report uses complete observations only. Rows with missing values in `Cig`, `T`, `TCI`, `GDP`, `country`, or `year` are removed before estimation.
