
# Panel regression report for presentation
# Model: Cig_it = alpha + beta*T_it + gamma*TCI_it + delta*GDP_it + country FE + year FE + error_it
# Output: index.html, a clean HTML report that you can publish with GitHub Pages and link from PowerPoint/PDF.

from pathlib import Path
import base64
import io

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

DATA_FILE = Path("Data17.xlsx")      # Put the Excel file in the same folder as this script
SHEET_NAME = "Sheet1"
OUTPUT_HTML = Path("index.html")     # GitHub Pages automatically opens this file


def clean_data(path: Path = DATA_FILE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read and clean the Excel data."""
    raw = pd.read_excel(path, sheet_name=SHEET_NAME)
    raw.columns = [str(c).strip() for c in raw.columns]

    required = ["country", "year", "GDP", "Cig", "TCI", "T"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = raw.copy()
    df["country"] = df["country"].astype(str).str.strip()
    for col in ["year", "GDP", "Cig", "TCI", "T"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    model_df = df.dropna(subset=["country", "year", "GDP", "Cig", "TCI", "T"]).copy()
    model_df["year"] = model_df["year"].astype(int).astype(str)

    # Rename T because T can sometimes conflict with formula syntax.
    model_df = model_df.rename(columns={"T": "tax"})
    return df, model_df


def run_model(model_df: pd.DataFrame):
    """Two-way fixed effects using country and year dummy variables."""
    formula = "Cig ~ tax + TCI + GDP + C(country) + C(year)"
    model = smf.ols(formula, data=model_df).fit(
        cov_type="cluster",
        cov_kwds={"groups": model_df["country"]}
    )
    return model


def fig_to_base64(fig) -> str:
    """Convert a matplotlib figure into a base64 PNG for embedding in HTML."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def make_charts(model_df: pd.DataFrame, model) -> dict[str, str]:
    """Create simple presentation-ready charts."""
    charts = {}

    by_year = model_df.assign(year_int=model_df["year"].astype(int)).groupby("year_int", as_index=False).agg(
        avg_cig=("Cig", "mean"),
        avg_tax=("tax", "mean"),
        avg_tci=("TCI", "mean"),
        avg_gdp=("GDP", "mean"),
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(by_year["year_int"], by_year["avg_cig"], marker="o")
    ax.set_title("Average cigarette consumption over time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Cig")
    ax.grid(True, alpha=0.3)
    charts["avg_cig"] = fig_to_base64(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(by_year["year_int"], by_year["avg_tax"], marker="o")
    ax.set_title("Average tax variable over time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average T")
    ax.grid(True, alpha=0.3)
    charts["avg_tax"] = fig_to_base64(fig)

    fitted = model.fittedvalues
    residuals = model.resid

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(model_df["Cig"], fitted, alpha=0.7)
    low = min(model_df["Cig"].min(), fitted.min())
    high = max(model_df["Cig"].max(), fitted.max())
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_title("Actual vs fitted values")
    ax.set_xlabel("Actual Cig")
    ax.set_ylabel("Fitted Cig")
    ax.grid(True, alpha=0.3)
    charts["actual_fitted"] = fig_to_base64(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=20)
    ax.set_title("Distribution of residuals")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    charts["residuals"] = fig_to_base64(fig)

    return charts


def make_html(raw: pd.DataFrame, model_df: pd.DataFrame, model, charts: dict[str, str]) -> str:
    """Build a single self-contained HTML file."""
    main_vars = ["tax", "TCI", "GDP"]
    coef_table = pd.DataFrame({
        "Variable": ["T / tax", "TCI", "GDP"],
        "Coefficient": [model.params[v] for v in main_vars],
        "Std. Error": [model.bse[v] for v in main_vars],
        "p-value": [model.pvalues[v] for v in main_vars],
        "95% CI low": [model.conf_int().loc[v, 0] for v in main_vars],
        "95% CI high": [model.conf_int().loc[v, 1] for v in main_vars],
    })

    for col in ["Coefficient", "Std. Error", "p-value", "95% CI low", "95% CI high"]:
        coef_table[col] = coef_table[col].map(lambda x: f"{x:.4f}")

    desc = model_df[["Cig", "tax", "TCI", "GDP"]].describe().round(3)

    n_raw = len(raw)
    n_model = len(model_df)
    countries = model_df["country"].nunique()
    years = model_df["year"].nunique()
    missing_cig = raw["Cig"].isna().sum()

    # A simple interpretation sentence, using cautious language.
    tax_coef = model.params["tax"]
    tax_p = model.pvalues["tax"]
    tax_text = (
        f"The estimated coefficient for T is {tax_coef:.3f}. "
        f"Holding GDP, TCI, country fixed effects and year fixed effects constant, "
        f"a one-unit increase in T is associated with about {tax_coef:.3f} units change in Cig. "
        f"The p-value is {tax_p:.3f}. This is an association, not automatic proof of causality."
    )

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Panel Regression Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; background: #f6f8fa; color: #1f2937; }}
        header {{ background: #0f766e; color: white; padding: 32px 44px; }}
        header h1 {{ margin: 0; font-size: 32px; }}
        header p {{ margin: 8px 0 0 0; font-size: 17px; }}
        main {{ max-width: 1100px; margin: 24px auto; padding: 0 24px 60px 24px; }}
        section {{ background: white; border-radius: 14px; padding: 24px; margin-bottom: 22px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); }}
        h2 {{ color: #0f766e; margin-top: 0; }}
        .equation {{ font-size: 22px; padding: 18px; background: #eefdf9; border-left: 5px solid #0f766e; border-radius: 8px; overflow-x: auto; }}
        .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; }}
        .card {{ background: #f9fafb; padding: 16px; border-radius: 10px; border: 1px solid #e5e7eb; }}
        .card .num {{ font-size: 28px; font-weight: bold; color: #0f766e; }}
        table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
        th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; text-align: right; }}
        th:first-child, td:first-child {{ text-align: left; }}
        th {{ background: #f3f4f6; }}
        .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }}
        .chart {{ width: 100%; border: 1px solid #e5e7eb; border-radius: 10px; }}
        .note {{ background: #fff7ed; border-left: 5px solid #f97316; padding: 14px; border-radius: 8px; }}
        details {{ margin-top: 12px; }}
        summary {{ cursor: pointer; font-weight: bold; color: #0f766e; }}
        footer {{ text-align: center; color: #6b7280; padding: 24px; }}
        code {{ background: #f3f4f6; padding: 2px 5px; border-radius: 4px; }}
    </style>
</head>
<body>
<header>
    <h1>Panel Regression Results</h1>
    <p>Two-way fixed effects model using country and year effects</p>
</header>
<main>
    <section>
        <h2>Model</h2>
        <div class="equation">Cig<sub>it</sub> = α + βT<sub>it</sub> + γTCI<sub>it</sub> + δGDP<sub>it</sub> + μ<sub>i</sub> + λ<sub>t</sub> + ε<sub>it</sub></div>
        <p><strong>Meaning:</strong> <code>Cig</code> is the dependent variable. <code>T</code>, <code>TCI</code>, and <code>GDP</code> are explanatory variables. <code>μ_i</code> means country fixed effects and <code>λ_t</code> means year fixed effects.</p>
    </section>

    <section>
        <h2>Data used in the model</h2>
        <div class="cards">
            <div class="card"><div class="num">{n_raw}</div><div>Raw rows</div></div>
            <div class="card"><div class="num">{n_model}</div><div>Complete model observations</div></div>
            <div class="card"><div class="num">{countries}</div><div>Countries</div></div>
            <div class="card"><div class="num">{years}</div><div>Years</div></div>
            <div class="card"><div class="num">{missing_cig}</div><div>Missing Cig values</div></div>
        </div>
        <p class="note">Rows with missing values in <code>Cig</code>, <code>T</code>, <code>TCI</code>, <code>GDP</code>, <code>country</code>, or <code>year</code> were removed before estimation.</p>
    </section>

    <section>
        <h2>Main regression coefficients</h2>
        {coef_table.to_html(index=False, escape=False)}
        <p>{tax_text}</p>
    </section>

    <section>
        <h2>Model fit</h2>
        <div class="cards">
            <div class="card"><div class="num">{model.rsquared:.3f}</div><div>R-squared</div></div>
            <div class="card"><div class="num">{model.rsquared_adj:.3f}</div><div>Adjusted R-squared</div></div>
            <div class="card"><div class="num">{int(model.nobs)}</div><div>Observations</div></div>
        </div>
    </section>

    <section>
        <h2>Charts</h2>
        <div class="chart-grid">
            <img class="chart" src="data:image/png;base64,{charts['avg_cig']}" alt="Average Cig over time">
            <img class="chart" src="data:image/png;base64,{charts['avg_tax']}" alt="Average tax over time">
            <img class="chart" src="data:image/png;base64,{charts['actual_fitted']}" alt="Actual vs fitted">
            <img class="chart" src="data:image/png;base64,{charts['residuals']}" alt="Residuals">
        </div>
    </section>

    <section>
        <h2>Descriptive statistics</h2>
        {desc.to_html(escape=False)}
    </section>

    <section>
        <h2>Full regression output</h2>
        <details>
            <summary>Click to show full statsmodels table</summary>
            {model.summary().as_html()}
        </details>
    </section>
</main>
<footer>
    Generated with Python: pandas, statsmodels and matplotlib.
</footer>
</body>
</html>
"""
    return html


def main():
    raw, model_df = clean_data(DATA_FILE)
    model = run_model(model_df)
    charts = make_charts(model_df, model)
    html = make_html(raw, model_df, model, charts)
    OUTPUT_HTML.write_text(html, encoding="utf-8")

    print("Report created:", OUTPUT_HTML.resolve())
    print("Main coefficients:")
    print(model.params[["tax", "TCI", "GDP"]])
    print("p-values:")
    print(model.pvalues[["tax", "TCI", "GDP"]])


if __name__ == "__main__":
    main()
