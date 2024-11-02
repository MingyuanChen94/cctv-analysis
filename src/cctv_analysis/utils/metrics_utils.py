import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from .logging_utils import setup_logging

logger = setup_logging(__name__)


class MetricsCalculator:
    """Calculates various metrics from CCTV analysis data."""

    def __init__(self):
        """Initialize metrics calculator."""
        pass

    def calculate_all_metrics(self, data: pd.DataFrame) -> Dict:
        """
        Calculate all available metrics.

        Args:
            data (pd.DataFrame): Detection and tracking data

        Returns:
            Dict: Dictionary containing all calculated metrics
        """
        metrics = {}

        # Basic metrics
        metrics.update(self.calculate_basic_metrics(data))

        # Temporal metrics
        metrics.update(self.calculate_temporal_metrics(data))

        # Demographic metrics
        metrics.update(self.calculate_demographic_metrics(data))

        # Movement metrics
        metrics.update(self.calculate_movement_metrics(data))

        # Dwell time metrics
        metrics.update(self.calculate_dwell_time_metrics(data))

        return metrics

    def calculate_basic_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate basic counting metrics."""
        metrics = {
            "total_unique_individuals": len(data),
            "camera_coverage": {
                "camera1_only": len(
                    data[
                        (data["appear_in_camera_1"] == 1)
                        & (data["appear_in_camera_2"] == 0)
                    ]
                ),
                "camera2_only": len(
                    data[
                        (data["appear_in_camera_1"] == 0)
                        & (data["appear_in_camera_2"] == 1)
                    ]
                ),
                "both_cameras": len(
                    data[
                        (data["appear_in_camera_1"] == 1)
                        & (data["appear_in_camera_2"] == 1)
                    ]
                ),
            },
            "camera_utilization": {
                "camera1_total": data["appear_in_camera_1"].sum(),
                "camera2_total": data["appear_in_camera_2"].sum(),
            },
        }

        # Calculate percentages
        total = len(data)
        for key, value in metrics["camera_coverage"].items():
            metrics["camera_coverage"][f"{key}_percentage"] = (
                (value / total * 100) if total > 0 else 0
            )

        return metrics

    def calculate_temporal_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate time-based metrics."""
        metrics = {
            "temporal_analysis": {
                "peak_hours": {},
                "average_daily_count": {},
                "time_distribution": {},
            }
        }

        # Process camera 1 data
        camera1_data = data[data["appear_in_camera_1"] == 1]
        if not camera1_data.empty:
            metrics["temporal_analysis"].update(
                {
                    "camera1": self._analyze_temporal_patterns(
                        camera1_data, "first_seen_camera1"
                    )
                }
            )

        # Process camera 2 data
        camera2_data = data[data["appear_in_camera_2"] == 1]
        if not camera2_data.empty:
            metrics["temporal_analysis"].update(
                {
                    "camera2": self._analyze_temporal_patterns(
                        camera2_data, "first_seen_camera2"
                    )
                }
            )

        return metrics

    def _analyze_temporal_patterns(
        self, data: pd.DataFrame, timestamp_column: str
    ) -> Dict:
        """Analyze temporal patterns in the data."""
        results = {}

        # Convert timestamps to datetime if they aren't already
        if not pd.api.types.is_datetime64_any_dtype(data[timestamp_column]):
            data[timestamp_column] = pd.to_datetime(data[timestamp_column])

        # Group by hour
        hourly_counts = data[timestamp_column].dt.hour.value_counts().sort_index()

        # Find peak hours (top 3)
        peak_hours = hourly_counts.nlargest(3)
        results["peak_hours"] = peak_hours.to_dict()

        # Calculate average counts by hour
        results["hourly_average"] = hourly_counts.mean()

        # Calculate day of week distribution
        dow_counts = data[timestamp_column].dt.day_name().value_counts()
        results["day_of_week_distribution"] = dow_counts.to_dict()

        return results

    def calculate_demographic_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate demographic-related metrics."""
        metrics = {
            "demographics": {
                "age_distribution": {},
                "gender_distribution": {},
                "cross_analysis": {},
            }
        }

        # Age distribution
        age_counts = data["age_group"].value_counts()
        metrics["demographics"]["age_distribution"] = {
            "counts": age_counts.to_dict(),
            "percentages": (age_counts / len(data) * 100).to_dict(),
        }

        # Gender distribution
        gender_counts = data["gender"].value_counts()
        metrics["demographics"]["gender_distribution"] = {
            "counts": gender_counts.to_dict(),
            "percentages": (gender_counts / len(data) * 100).to_dict(),
        }

        # Cross-analysis of age and gender
        cross_tab = pd.crosstab(data["age_group"], data["gender"])
        metrics["demographics"]["cross_analysis"] = {
            "age_gender_matrix": cross_tab.to_dict()
        }

        return metrics

    def calculate_movement_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate movement and transition metrics."""
        metrics = {
            "movement_patterns": {
                "transitions": {},
                "flow_analysis": {},
                "path_metrics": {},
            }
        }

        # Calculate transition times for those appearing in both cameras
        transition_data = data[data["appear_camera1_then_camera2"] == 1]
        if not transition_data.empty:
            transition_times = transition_data.apply(
                lambda x: (
                    x["first_seen_camera2"] - x["last_seen_camera1"]
                ).total_seconds(),
                axis=1,
            )

            metrics["movement_patterns"]["transitions"] = {
                "average_transition_time": transition_times.mean(),
                "median_transition_time": transition_times.median(),
                "min_transition_time": transition_times.min(),
                "max_transition_time": transition_times.max(),
                "std_transition_time": transition_times.std(),
            }

        # Flow analysis
        metrics["movement_patterns"]["flow_analysis"] = {
            "camera1_to_camera2_ratio": (
                len(transition_data) / len(data[data["appear_in_camera_1"] == 1])
                if len(data[data["appear_in_camera_1"] == 1]) > 0
                else 0
            ),
            "bidirectional_movement_count": len(
                data[
                    (data["appear_in_camera_1"] == 1)
                    & (data["appear_in_camera_2"] == 1)
                ]
            ),
        }

        return metrics

    def calculate_dwell_time_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate dwell time related metrics."""
        metrics = {"dwell_time": {"camera1": {}, "camera2": {}}}

        # Camera 1 dwell times
        camera1_dwell = data[data["appear_in_camera_1"] == 1]["dwell_time_camera1"]
        if not camera1_dwell.empty:
            metrics["dwell_time"]["camera1"] = self._calculate_dwell_statistics(
                camera1_dwell
            )

        # Camera 2 dwell times
        camera2_dwell = data[data["appear_in_camera_2"] == 1]["dwell_time_camera2"]
        if not camera2_dwell.empty:
            metrics["dwell_time"]["camera2"] = self._calculate_dwell_statistics(
                camera2_dwell
            )

        return metrics

    def _calculate_dwell_statistics(self, dwell_times: pd.Series) -> Dict:
        """Calculate statistics for dwell times."""
        return {
            "average": dwell_times.mean(),
            "median": dwell_times.median(),
            "std": dwell_times.std(),
            "min": dwell_times.min(),
            "max": dwell_times.max(),
            "percentiles": {
                "25th": dwell_times.quantile(0.25),
                "75th": dwell_times.quantile(0.75),
                "90th": dwell_times.quantile(0.90),
            },
        }

    def calculate_correlation_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate correlation metrics between different variables."""
        metrics = {"correlations": {}}

        # Select numerical columns
        numerical_cols = [
            "dwell_time_camera1",
            "dwell_time_camera2",
            "appear_camera1_then_camera2",
        ]

        # Calculate correlation matrix
        correlation_matrix = data[numerical_cols].corr()
        metrics["correlations"]["matrix"] = correlation_matrix.to_dict()

        return metrics

    def get_statistical_significance(
        self, data: pd.DataFrame, alpha: float = 0.05
    ) -> Dict:
        """
        Perform statistical significance tests on various metrics.

        Args:
            data (pd.DataFrame): Detection data
            alpha (float): Significance level

        Returns:
            Dict: Statistical test results
        """
        results = {"statistical_tests": {}}

        # Test for gender differences in dwell time
        for camera in ["camera1", "camera2"]:
            dwell_col = f"dwell_time_{camera}"
            if dwell_col in data.columns:
                male_dwell = data[data["gender"] == "Male"][dwell_col].dropna()
                female_dwell = data[data["gender"] == "Female"][dwell_col].dropna()

                if len(male_dwell) > 0 and len(female_dwell) > 0:
                    t_stat, p_value = stats.ttest_ind(male_dwell, female_dwell)
                    results["statistical_tests"][f"{camera}_gender_dwell_time"] = {
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "significant": p_value < alpha,
                    }

        # Test for age group differences
        age_groups = data["age_group"].unique()
        if len(age_groups) > 1:
            for camera in ["camera1", "camera2"]:
                dwell_col = f"dwell_time_{camera}"
                if dwell_col in data.columns:
                    age_group_data = [
                        data[data["age_group"] == group][dwell_col].dropna()
                        for group in age_groups
                    ]

                    if all(len(group) > 0 for group in age_group_data):
                        f_stat, p_value = stats.f_oneway(*age_group_data)
                        results["statistical_tests"][
                            f"{camera}_age_group_dwell_time"
                        ] = {
                            "f_statistic": f_stat,
                            "p_value": p_value,
                            "significant": p_value < alpha,
                        }

        return results


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 100

    # Generate timestamps
    base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    camera1_times = [
        base_time + timedelta(minutes=np.random.randint(0, 1440))
        for _ in range(n_samples)
    ]
    camera2_times = [
        t + timedelta(minutes=np.random.randint(5, 30)) for t in camera1_times
    ]

    # Create DataFrame
    data = pd.DataFrame(
        {
            "individual_id": range(n_samples),
            "age_group": np.random.choice(["0-18", "19-35", "36-50", "51+"], n_samples),
            "gender": np.random.choice(["Male", "Female"], n_samples),
            "appear_in_camera_1": np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
            "appear_in_camera_2": np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            "appear_camera1_then_camera2": np.random.choice(
                [0, 1], n_samples, p=[0.4, 0.6]
            ),
            "first_seen_camera1": camera1_times,
            "last_seen_camera1": [
                t + timedelta(minutes=np.random.randint(1, 10)) for t in camera1_times
            ],
            "first_seen_camera2": camera2_times,
            "last_seen_camera2": [
                t + timedelta(minutes=np.random.randint(1, 10)) for t in camera2_times
            ],
        }
    )

    # Calculate dwell times
    data["dwell_time_camera1"] = data.apply(
        lambda x: (x["last_seen_camera1"] - x["first_seen_camera1"]).total_seconds()
        if x["appear_in_camera_1"] == 1
        else np.nan,
        axis=1,
    )

    data["dwell_time_camera2"] = data.apply(
        lambda x: (x["last_seen_camera2"] - x["first_seen_camera2"]).total_seconds()
        if x["appear_in_camera_2"] == 1
        else np.nan,
        axis=1,
    )

    # Calculate metrics
    calculator = MetricsCalculator()
    metrics = calculator.calculate_all_metrics(data)

    # Print some example metrics
    print("\nBasic Metrics:")
    print(f"Total unique individuals: {metrics['total_unique_individuals']}")
    print("\nCamera Coverage:")
    print(metrics["camera_coverage"])

    print("\nDwell Time Metrics:")
    print(metrics["dwell_time"])

    # Calculate statistical significance
    stats_results = calculator.get_statistical_significance(data)
    print("\nStatistical Test Results:")
    print(stats_results["statistical_tests"])
