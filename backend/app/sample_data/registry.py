"""
Sample dataset registry (session 14, Demo Gallery).

Three bundled datasets, synthetically generated but deliberately containing
real, demonstrable data-quality issues (missing values, category-name
variants, genuine outliers, exact duplicates) — the point is for a visitor
to see the cleaning/quality engines actually find something, not to load a
suspiciously perfect dataset. Each ships with a canned `investigation_goal`
so goal-relevance threading (session 3) has something to demonstrate too.
"""
from pathlib import Path

SAMPLE_DATA_DIR = Path(__file__).resolve().parent
EXAMPLE_REPORTS_DIR = SAMPLE_DATA_DIR / "example_reports"

SAMPLES = [
    {
        "id": "retail_sales",
        "name": "Retail Sales",
        "description": "E-commerce order-level sales across 4 regions and 5 product categories. "
                        "Has missing prices, category-name inconsistencies (e.g. 'Electronics' vs "
                        "'electronics'), a handful of clear data-entry-error outliers, and a few "
                        "exact duplicate rows.",
        "filename": "retail_sales.csv",
        "investigation_goal": "find which regions and categories drive the most revenue",
        "highlights": ["Category name merging", "Missing value imputation", "Outlier detection"],
    },
    {
        "id": "employee_attrition",
        "name": "Employee Attrition",
        "description": "HR data on 400+ employees across departments and roles, with attrition "
                        "outcome. Has missing satisfaction scores, department-name variants, and a "
                        "few unrealistic income outliers.",
        "filename": "employee_attrition.csv",
        "investigation_goal": "understand what's associated with employee attrition",
        "highlights": ["Hypothesis testing", "Clustering", "Category cleanup"],
    },
    {
        "id": "website_traffic",
        "name": "Website Traffic",
        "description": "A full year of daily session/conversion metrics with weekly seasonality. "
                        "Has missing bounce-rate values and a few obvious tracking-glitch spikes in "
                        "session counts.",
        "filename": "website_traffic.csv",
        "investigation_goal": "understand the trend in sessions over the year and forecast ahead",
        "highlights": ["Trend analysis", "Forecasting", "Time-series EDA"],
    },
]

SAMPLES_BY_ID = {s["id"]: s for s in SAMPLES}


def get_sample_path(sample_id: str) -> Path:
    return SAMPLE_DATA_DIR / SAMPLES_BY_ID[sample_id]["filename"]


def get_example_report_path(sample_id: str) -> Path:
    return EXAMPLE_REPORTS_DIR / f"{sample_id}.html"
