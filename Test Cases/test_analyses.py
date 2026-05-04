
import os

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan

# Helpers replicated from the analysis scripts

street_keywords = [
    "street", "Street", "Monaco", "Marina Bay", "Baku", "Albert Park",
    "Adelaide", "Phoenix", "Detroit", "Long Beach", "Las Vegas",
    "Miami", "Jeddah", "Corniche", "Valencia Street", "Montjuïc",
    "Boavista", "AVUS", "Gilles Villeneuve",
]


def era_bucket(y):
    """Map a calendar year to its F1 era label."""
    if y <= 2009:
        return "2003-2009"
    if y <= 2019:
        return "2010-2019"
    return "2020-2022"


def is_street(name):
    """Return True if the circuit name matches any street-circuit keyword."""
    if pd.isna(name):
        return False
    return any(kw in str(name) for kw in street_keywords)


def sig_stars(p):
    """Return significance stars for a given p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"



@pytest.fixture(scope="module")
def synthetic_ols_df():
    """50-row synthetic DataFrame suitable for the OLS model."""
    rng = np.random.default_rng(0)
    n = 50
    grid = rng.integers(1, 21, size=n).astype(float)
    era_2010_2019 = rng.integers(0, 2, size=n)
    # ensure era dummies don't overlap
    era_2020_2022 = (1 - era_2010_2019) * rng.integers(0, 2, size=n)
    circuit_type_street = rng.integers(0, 2, size=n)
    # position correlated with grid so the OLS coefficient is clearly positive
    position = np.clip(np.round(1.5 * grid + rng.normal(0, 3, size=n)), 1, 20)
    return pd.DataFrame({
        "position": position,
        "grid": grid,
        "era_2010_2019": era_2010_2019,
        "era_2020_2022": era_2020_2022,
        "circuit_type_street": circuit_type_street,
    })


@pytest.fixture(scope="module")
def ols_result(synthetic_ols_df):
    """Fitted OLS result on the synthetic DataFrame."""
    df = synthetic_ols_df
    X = sm.add_constant(
        df[["grid", "era_2010_2019", "era_2020_2022", "circuit_type_street"]]
    )
    y = df["position"]
    return sm.OLS(y, X).fit()


@pytest.fixture(scope="module")
def synthetic_logit_df():
    """200-row synthetic DataFrame for logistic regression (mostly 0s, a few 1s)."""
    rng = np.random.default_rng(1)
    n = 200
    pole = rng.integers(0, 2, size=n)
    era_2010_2019 = rng.integers(0, 2, size=n)
    era_2020_2022 = (1 - era_2010_2019) * rng.integers(0, 2, size=n)
    circuit_type_street = rng.integers(0, 2, size=n)
    # pole strongly predicts win so the pole coefficient is clearly positive
    log_odds = -3.0 + 2.5 * pole
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    win = rng.binomial(1, prob)
    return pd.DataFrame({
        "win": win,
        "pole": pole,
        "era_2010_2019": era_2010_2019,
        "era_2020_2022": era_2020_2022,
        "circuit_type_street": circuit_type_street,
    })


@pytest.fixture(scope="module")
def logit_result(synthetic_logit_df):
    """Fitted Logit result on the synthetic DataFrame."""
    df = synthetic_logit_df
    X = sm.add_constant(
        df[["pole", "era_2010_2019", "era_2020_2022", "circuit_type_street"]]
    )
    y = df["win"]
    return sm.Logit(y, X).fit(maxiter=200, disp=False)


# Group 1: analysis_linear_regression.py test

class TestLinearRegression:

    def test_era_bucket_mapping(self):
        """era_bucket returns the correct era string for every year in each period."""
        for yr in range(2003, 2010):
            assert era_bucket(yr) == "2003-2009", f"Failed for year {yr}"
        for yr in range(2010, 2020):
            assert era_bucket(yr) == "2010-2019", f"Failed for year {yr}"
        for yr in range(2020, 2023):
            assert era_bucket(yr) == "2020-2022", f"Failed for year {yr}"

    def test_street_circuit_classification(self):
        """is_street is True for street circuits and False for permanent ones."""
        for name in ["Monaco", "Marina Bay Street Circuit", "Baku City Circuit"]:
            assert is_street(name) is True, f"Expected True for '{name}'"
        for name in ["Silverstone", "Spa-Francorchamps", "Suzuka"]:
            assert is_street(name) is False, f"Expected False for '{name}'"

    def test_dummy_variable_encoding(self):
        """Dummy variables encode era and circuit_type columns correctly."""
        df = pd.DataFrame({
            "year": [2005, 2015, 2021],
            "circuit_type": ["Permanent", "Street", "Permanent"],
        })
        df["era"] = df["year"].apply(era_bucket)
        df["era_2010_2019"] = (df["era"] == "2010-2019").astype(int)
        df["era_2020_2022"] = (df["era"] == "2020-2022").astype(int)
        df["circuit_type_street"] = (df["circuit_type"] == "Street").astype(int)

        # era_2010_2019 is 1 only for 2015
        assert list(df["era_2010_2019"]) == [0, 1, 0]
        # era_2020_2022 is 1 only for 2021
        assert list(df["era_2020_2022"]) == [0, 0, 1]
        # circuit_type_street is 1 only for the Street row (2015)
        assert list(df["circuit_type_street"]) == [0, 1, 0]

    def test_ols_model_runs_on_synthetic_data(self, ols_result):
        """OLS model fits successfully and has expected coefficient properties."""
        result = ols_result
        assert result is not None
        assert 0.0 <= result.rsquared <= 1.0
        assert result.params["grid"] > 0, "grid coefficient should be positive"
        expected = {"const", "grid", "era_2010_2019", "era_2020_2022", "circuit_type_street"}
        assert expected == set(result.params.index)

    def test_year_filter_bounds(self):
        """Year filter (>=2003, <=2022) excludes 2002 and 2023."""
        df = pd.DataFrame({"year": [2002, 2003, 2015, 2022, 2023]})
        filtered = df[(df["year"] >= 2003) & (df["year"] <= 2022)]
        assert 2002 not in filtered["year"].values
        assert 2023 not in filtered["year"].values
        assert set(filtered["year"].values) == {2003, 2015, 2022}


# Group 2: analysis_residual_plots.py test

class TestResidualDiagnostics:

    def test_breusch_pagan_returns_four_values(self, ols_result):
        """het_breuschpagan returns exactly 4 numeric values."""
        result = ols_result
        bp_vals = het_breuschpagan(result.resid, result.model.exog)
        assert len(bp_vals) == 4
        for val in bp_vals:
            assert isinstance(val, (float, int, np.floating)), (
                f"Expected numeric, got {type(val)}: {val}"
            )

    def test_shapiro_wilk_on_normal_data(self):
        """Shapiro-Wilk p-value > 0.05 for 500 samples drawn from a normal distribution."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, size=500)
        _, p_value = stats.shapiro(data)
        assert p_value > 0.05, f"Expected p > 0.05 for normal data, got {p_value:.4f}"

    def test_shapiro_wilk_on_skewed_data(self):
        """Shapiro-Wilk p-value < 0.05 for 500 samples from an exponential distribution."""
        rng = np.random.default_rng(42)
        data = rng.exponential(scale=1.0, size=500)
        _, p_value = stats.shapiro(data)
        assert p_value < 0.05, f"Expected p < 0.05 for exponential data, got {p_value:.4f}"

    def test_residual_computation(self, ols_result):
        """result.resid equals y - result.fittedvalues element-wise."""
        result = ols_result
        expected_resid = result.model.endog - result.fittedvalues.values
        np.testing.assert_allclose(result.resid.values, expected_resid, rtol=1e-10)

    def test_studentized_residuals_shape(self, ols_result):
        """Studentized residuals array has the same length as the number of observations."""
        result = ols_result
        std_resid = result.get_influence().resid_studentized_internal
        assert len(std_resid) == int(result.nobs)


# analysis_logistic_regression.py test

class TestLogisticRegression:

    def test_win_and_pole_binary_encoding(self):
        """win==1 iff position==1; pole==1 iff grid==1."""
        df = pd.DataFrame({
            "position": [1, 3, 1, 10],
            "grid":     [1, 1, 5,  5],
        })
        df["win"]  = (df["position"] == 1).astype(int)
        df["pole"] = (df["grid"]     == 1).astype(int)

        assert list(df["win"])  == [1, 0, 1, 0]
        assert list(df["pole"]) == [1, 1, 0, 0]

    def test_logit_model_runs_on_synthetic_data(self, logit_result):
        """Logit model converges, pseudo-R² is in [0,1], and pole coef is positive."""
        result = logit_result
        assert result.mle_retvals["converged"] is True
        assert 0.0 <= result.prsquared <= 1.0
        assert result.params["pole"] > 0, "pole coefficient should be positive"
        expected = {"const", "pole", "era_2010_2019", "era_2020_2022", "circuit_type_street"}
        assert expected == set(result.params.index)

    def test_odds_ratio_calculation(self):
        """np.exp(log_odds) matches the expected odds ratio within 0.1% tolerance."""
        log_odds = 3.7274
        expected_or = 41.57
        np.testing.assert_allclose(np.exp(log_odds), expected_or, rtol=1e-3)

    def test_odds_ratio_ci_ordering(self, logit_result):
        """For every predictor, lower CI bound ≤ OR ≤ upper CI bound."""
        result = logit_result
        or_vals = np.exp(result.params)
        ci = np.exp(result.conf_int())
        for name in result.params.index:
            lo = ci.loc[name, 0]
            hi = ci.loc[name, 1]
            or_val = or_vals[name]
            assert lo <= or_val + 1e-10, f"OR < lower CI for '{name}'"
            assert or_val <= hi + 1e-10, f"OR > upper CI for '{name}'"

    def test_significance_star_function(self):
        """sig_stars returns the correct label for representative p-values."""
        assert sig_stars(0.0001) == "***"
        assert sig_stars(0.005)  == "**"
        assert sig_stars(0.03)   == "*"
        assert sig_stars(0.10)   == "ns"


#Cross cutting


class TestCrossCutting:

    def test_model_columns_no_nulls_after_dropna(self):
        """After dropna on model columns, no NaN values remain."""
        df = pd.DataFrame({
            "position":            [1.0, np.nan, 3.0, 4.0],
            "grid":                [1.0, 2.0,    np.nan, 4.0],
            "era_2010_2019":       [1,   0,      1,   0],
            "era_2020_2022":       [0,   1,      0,   1],
            "circuit_type_street": [1,   0,      0,   1],
        })
        model_cols = [
            "position", "grid", "era_2010_2019", "era_2020_2022", "circuit_type_street"
        ]
        model_df = df[model_cols].dropna()
        assert model_df.isnull().sum().sum() == 0

    def test_street_keywords_list_nonempty(self):
        """street_keywords contains at least 10 entries."""
        assert len(street_keywords) >= 10, (
            f"Expected ≥10 keywords, found {len(street_keywords)}"
        )

    def test_output_paths_use_project_root(self):
        """out_txt and out_plot are built from PROJECT_ROOT and have correct extensions."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(script_dir)

        out_txt  = os.path.join(PROJECT_ROOT, "plots", "analysis1_linear_regression_summary.txt")
        out_plot = os.path.join(PROJECT_ROOT, "plots", "analysis1_linear_regression.png")

        assert out_txt.endswith(".txt"),   "out_txt must end with .txt"
        assert out_plot.endswith(".png"),  "out_plot must end with .png"
        assert out_txt.startswith(PROJECT_ROOT),  "out_txt must be under PROJECT_ROOT"
        assert out_plot.startswith(PROJECT_ROOT), "out_plot must be under PROJECT_ROOT"
