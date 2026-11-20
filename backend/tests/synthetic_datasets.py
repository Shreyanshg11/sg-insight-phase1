"""
Synthetic multi-dataset fixtures for regression testing.

Problem this addresses: every engine in this app was validated during
development against one real dataset (an IPL cricket CSV). The engines
themselves are dataset-agnostic (no hardcoded column/dataset checks — see
the other engines' docstrings), but "dataset-agnostic in code" and
"actually behaves sanely across genuinely different data shapes" are
different claims, and only the second one is worth trusting without
evidence.

Each function below builds one small, self-contained synthetic dataset
shaped like a real-world domain (sales, HR, healthcare, real estate, IoT,
finance, survey data, etc.), each with a *different* mix of: missing-value
patterns (including a couple of deliberately structural ones, to stress
missingness_engine outside the cricket case it was designed against),
outliers, duplicate rows, categorical inconsistency, and column-type
composition (some are numeric-heavy, some categorical-heavy, some
timeseries-shaped).

These are synthetic, not real records — generated from named RandomState
seeds, no scraped or real personal/patient/employee data.
"""
import numpy as np
import pandas as pd


def make_sales_transactions_df(n=300, seed=10):
    rng = np.random.RandomState(seed)
    price = rng.normal(45, 12, n).round(2)
    quantity = rng.randint(1, 15, n)
    df = pd.DataFrame({
        "order_id": [f"ORD{i:05d}" for i in range(n)],
        "order_date": pd.date_range("2024-01-01", periods=n, freq="6h"),
        "price": price,
        "quantity": quantity,
        "revenue": price * quantity,
        "region": rng.choice(["North", "south", " North ", "East", "West"], n),
        "discount_pct": rng.choice([0, 5, 10, 15, np.nan], n, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
    })
    df.loc[df.sample(5, random_state=seed).index, "price"] = df["price"] * 50  # a few pricing errors
    return df


def make_hr_employee_df(n=250, seed=11):
    rng = np.random.RandomState(seed)
    tenure = rng.exponential(3, n).round(1)
    df = pd.DataFrame({
        "employee_id": [f"E{i:04d}" for i in range(n)],
        "hire_date": pd.date_range("2015-01-01", periods=n, freq="9D"),
        "salary": rng.normal(75000, 18000, n).round(0),
        "tenure_years": tenure,
        "department": rng.choice(["Engineering", "Sales", "Support", "Marketing", "Finance"], n),
        "performance_score": rng.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.15, 0.4, 0.3, 0.1]),
        "attrition": rng.choice([0, 1], n, p=[0.85, 0.15]),
    })
    return df


def make_healthcare_patient_df(n=220, seed=12):
    rng = np.random.RandomState(seed)
    age = rng.randint(18, 90, n)
    df = pd.DataFrame({
        "patient_id": [f"P{i:05d}" for i in range(n)],
        "admission_date": pd.date_range("2023-06-01", periods=n, freq="8h"),
        "age": age,
        "bmi": rng.normal(27, 5, n).round(1),
        "blood_pressure_systolic": rng.normal(125, 15, n).round(0),
        "diagnosis_code": rng.choice(["A01", "B22", "C14", "D07"], n),
        "length_of_stay_days": rng.exponential(3, n).round(1),
        "readmitted_30d": rng.choice([0, 1], n, p=[0.88, 0.12]),
    })
    # structural missingness: icu_days only populated when length_of_stay > 5 (icu is conditional on severity)
    icu = np.where(df["length_of_stay_days"] > 5, rng.exponential(2, n).round(1), np.nan)
    df["icu_days"] = icu
    return df


def make_real_estate_listings_df(n=200, seed=13):
    rng = np.random.RandomState(seed)
    has_pool = rng.choice([True, False], n, p=[0.25, 0.75])
    df = pd.DataFrame({
        "listing_id": [f"L{i:05d}" for i in range(n)],
        "price": rng.normal(420000, 120000, n).round(0),
        "sqft": rng.randint(600, 4500, n),
        "bedrooms": rng.randint(1, 6, n),
        "city": rng.choice(["Austin", "Denver", "Seattle", "Miami"], n),
        "property_type": rng.choice(["Single Family", "Condo", "Townhouse"], n),
        "has_pool": has_pool,
    })
    # structural missingness: pool_size_sqft only exists for listings that have a pool
    df["pool_size_sqft"] = np.where(has_pool, rng.randint(100, 600, n), np.nan)
    return df


def make_iot_sensor_readings_df(n=400, seed=14):
    rng = np.random.RandomState(seed)
    t = pd.date_range("2024-03-01", periods=n, freq="15min")
    temp = 20 + 5 * np.sin(np.linspace(0, 20 * np.pi, n)) + rng.normal(0, 0.5, n)
    df = pd.DataFrame({
        "timestamp": t,
        "sensor_id": rng.choice(["S1", "S2", "S3"], n),
        "temperature_c": temp.round(2),
        "humidity_pct": rng.normal(45, 8, n).round(1),
        "pressure_hpa": rng.normal(1013, 4, n).round(1),
    })
    spike_idx = df.sample(6, random_state=seed).index
    # A decimal-point sensor bug: 23.4C recorded as 234 (~10x median) rather
    # than an arbitrary sentinel value, so this is provably a magnitude-slip
    # error signature and not just "far from the fence".
    df.loc[spike_idx, "temperature_c"] = (df.loc[spike_idx, "temperature_c"].abs() + 1) * 10
    return df


def make_marketing_campaign_df(n=180, seed=15):
    rng = np.random.RandomState(seed)
    impressions = rng.randint(500, 50000, n)
    clicks = (impressions * rng.uniform(0.01, 0.08, n)).astype(int)
    conversions = (clicks * rng.uniform(0.02, 0.15, n)).astype(int)
    df = pd.DataFrame({
        "campaign_id": [f"C{i:04d}" for i in range(n)],
        "channel": rng.choice(["Search", "Social", "Email", "Display"], n),
        "spend": rng.normal(800, 300, n).round(2),
        "impressions": impressions,
        "clicks": clicks,
        "conversions": conversions,
        "start_date": pd.date_range("2024-02-01", periods=n, freq="2D"),
    })
    return df


def make_finance_stock_prices_df(n=260, seed=16):
    rng = np.random.RandomState(seed)
    price = 100 + np.cumsum(rng.normal(0, 1.5, n))
    df = pd.DataFrame({
        "trade_date": pd.date_range("2023-01-02", periods=n, freq="B"),
        "open": (price + rng.normal(0, 0.5, n)).round(2),
        "high": (price + np.abs(rng.normal(1, 0.5, n))).round(2),
        "low": (price - np.abs(rng.normal(1, 0.5, n))).round(2),
        "close": price.round(2),
        "volume": rng.randint(100000, 5000000, n),
    })
    return df


def make_survey_responses_df(n=300, seed=17):
    rng = np.random.RandomState(seed)
    owns_car = rng.choice([True, False], n, p=[0.6, 0.4])
    df = pd.DataFrame({
        "respondent_id": [f"R{i:04d}" for i in range(n)],
        "age": rng.randint(18, 75, n),
        "satisfaction_1_5": rng.choice([1, 2, 3, 4, 5], n),
        "would_recommend": rng.choice(["Yes", "No", "Maybe"], n),
        "owns_car": owns_car,
    })
    # structural missingness: car_brand only answered by respondents who own a car (skip-logic survey pattern)
    car_brand = pd.Series(rng.choice(["Toyota", "Honda", "Ford", "Tesla"], n), dtype=object)
    df["car_brand"] = car_brand.where(owns_car, np.nan)
    return df


def make_education_scores_df(n=210, seed=18):
    rng = np.random.RandomState(seed)
    study_hours = rng.exponential(4, n).round(1)
    df = pd.DataFrame({
        "student_id": [f"ST{i:04d}" for i in range(n)],
        "school": rng.choice(["Lincoln High", "Washington High", "Roosevelt High"], n),
        "grade_level": rng.choice([9, 10, 11, 12], n),
        "study_hours_per_week": study_hours,
        "test_score": (50 + study_hours * 4 + rng.normal(0, 8, n)).clip(0, 100).round(1),
    })
    return df


def make_inventory_supply_chain_df(n=190, seed=19):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "sku": [f"SKU{i:05d}" for i in range(n)],
        "warehouse": rng.choice(["W1", "W2", "W3"], n),
        "category": rng.choice(["Electronics", "Apparel", "Home", "Toys"], n),
        "stock_level": rng.randint(0, 5000, n),
        "lead_time_days": rng.exponential(7, n).round(1),
        "restock_date": pd.date_range("2024-01-01", periods=n, freq="4D"),
    })
    df.loc[df.sample(8, random_state=seed).index, "stock_level"] = np.nan
    return df


def make_energy_consumption_df(n=350, seed=20):
    rng = np.random.RandomState(seed)
    t = pd.date_range("2024-01-01", periods=n, freq="D")
    weekday = t.day_of_week < 5
    base = np.where(weekday, 40, 25)
    df = pd.DataFrame({
        "date": t,
        "kwh_used": (base + rng.normal(0, 4, n)).round(2),
        "avg_temp_c": rng.normal(15, 8, n).round(1),
        "building": rng.choice(["HQ", "Warehouse", "Retail"], n),
    })
    return df


def make_customer_churn_df(n=300, seed=21):
    rng = np.random.RandomState(seed)
    tenure = rng.randint(1, 72, n)
    df = pd.DataFrame({
        "customer_id": [f"CUST{i:05d}" for i in range(n)],
        "tenure_months": tenure,
        "monthly_charges": rng.normal(65, 20, n).round(2),
        "contract_type": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "churn": rng.choice([0, 1], n, p=[0.93, 0.07]),  # deliberately imbalanced, like real churn data
    })
    return df


def make_sports_player_stats_df(n=200, seed=22):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "player_id": [f"PL{i:04d}" for i in range(n)],
        "team": rng.choice(["Hawks", "Wolves", "Eagles", "Bears"], n),
        "position": rng.choice(["Guard", "Forward", "Center"], n),
        "points_per_game": rng.normal(12, 6, n).round(1),
        "rebounds_per_game": rng.exponential(4, n).round(1),
        "assists_per_game": rng.exponential(3, n).round(1),
    })
    return df


def make_web_traffic_timeseries_df(n=365, seed=23):
    rng = np.random.RandomState(seed)
    t = pd.date_range("2023-01-01", periods=n, freq="D")
    seasonal = 500 + 150 * np.sin(np.linspace(0, 4 * np.pi, n))
    trend = np.linspace(0, 300, n)
    df = pd.DataFrame({
        "date": t,
        "visits": (seasonal + trend + rng.normal(0, 30, n)).round(0).clip(min=0),
        "bounce_rate_pct": rng.normal(45, 8, n).clip(0, 100).round(1),
        "avg_session_seconds": rng.exponential(120, n).round(0),
    })
    return df


ALL_SYNTHETIC_DATASET_BUILDERS = {
    "sales_transactions": make_sales_transactions_df,
    "hr_employee": make_hr_employee_df,
    "healthcare_patient": make_healthcare_patient_df,
    "real_estate_listings": make_real_estate_listings_df,
    "iot_sensor_readings": make_iot_sensor_readings_df,
    "marketing_campaign": make_marketing_campaign_df,
    "finance_stock_prices": make_finance_stock_prices_df,
    "survey_responses": make_survey_responses_df,
    "education_scores": make_education_scores_df,
    "inventory_supply_chain": make_inventory_supply_chain_df,
    "energy_consumption": make_energy_consumption_df,
    "customer_churn": make_customer_churn_df,
    "sports_player_stats": make_sports_player_stats_df,
    "web_traffic_timeseries": make_web_traffic_timeseries_df,
}
