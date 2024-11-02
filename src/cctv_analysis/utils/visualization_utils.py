import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

from .logging_utils import setup_logging

logger = setup_logging(__name__)


class VisualizationManager:
    """Manages creation and saving of various visualizations."""

    def __init__(self, output_dir: str = "output/visualizations"):
        """
        Initialize visualization manager.

        Args:
            output_dir (str): Directory for saving visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set default style
        plt.style.use("seaborn")

    def plot_traffic_flow(
        self,
        data: pd.DataFrame,
        save_path: Optional[str] = None,
        interactive: bool = False,
    ):
        """
        Plot traffic flow patterns between cameras.

        Args:
            data (pd.DataFrame): Detection data
            save_path (Optional[str]): Path to save the plot
            interactive (bool): Whether to create interactive plot
        """
        if interactive:
            return self._plot_traffic_flow_interactive(data, save_path)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Camera 1 appearances over time
        camera1_data = data[data["appear_in_camera_1"] == 1]
        ax1.hist(camera1_data["first_seen_camera1"], bins=20, alpha=0.7)
        ax1.set_title("Traffic Flow - Camera 1")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Number of Individuals")

        # Camera 2 appearances over time
        camera2_data = data[data["appear_in_camera2"] == 1]
        ax2.hist(camera2_data["first_seen_camera2"], bins=20, alpha=0.7)
        ax2.set_title("Traffic Flow - Camera 2")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Number of Individuals")

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path)
            plt.close()

    def _plot_traffic_flow_interactive(
        self, data: pd.DataFrame, save_path: Optional[str] = None
    ):
        """Create interactive traffic flow visualization using plotly."""
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Traffic Flow - Camera 1", "Traffic Flow - Camera 2"),
        )

        # Camera 1 data
        camera1_data = data[data["appear_in_camera_1"] == 1]
        fig.add_trace(
            go.Histogram(x=camera1_data["first_seen_camera1"], name="Camera 1"),
            row=1,
            col=1,
        )

        # Camera 2 data
        camera2_data = data[data["appear_in_camera2"] == 1]
        fig.add_trace(
            go.Histogram(x=camera2_data["first_seen_camera2"], name="Camera 2"),
            row=2,
            col=1,
        )

        fig.update_layout(
            height=800, showlegend=True, title_text="Traffic Flow Analysis"
        )

        if save_path:
            fig.write_html(self.output_dir / save_path)

        return fig

    def plot_demographics(self, data: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create demographic visualization plots.

        Args:
            data (pd.DataFrame): Detection data
            save_path (Optional[str]): Path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Age distribution
        sns.histplot(data=data, x="age_group", ax=ax1)
        ax1.set_title("Age Distribution")
        ax1.set_xlabel("Age Group")
        ax1.set_ylabel("Count")

        # Gender distribution
        data["gender"].value_counts().plot(kind="pie", ax=ax2, autopct="%1.1f%%")
        ax2.set_title("Gender Distribution")

        # Age distribution by gender
        sns.boxplot(data=data, x="gender", y="age_group", ax=ax3)
        ax3.set_title("Age Distribution by Gender")

        # Camera appearances by demographic
        appearance_data = pd.DataFrame(
            {
                "Camera 1": data[data["appear_in_camera_1"] == 1][
                    "gender"
                ].value_counts(),
                "Camera 2": data[data["appear_in_camera_2"] == 1][
                    "gender"
                ].value_counts(),
            }
        )
        appearance_data.plot(kind="bar", ax=ax4)
        ax4.set_title("Camera Appearances by Gender")
        ax4.set_xlabel("Gender")
        ax4.set_ylabel("Count")

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path)
            plt.close()

    def plot_movement_patterns(
        self, data: pd.DataFrame, save_path: Optional[str] = None
    ):
        """
        Visualize movement patterns between cameras.

        Args:
            data (pd.DataFrame): Detection data
            save_path (Optional[str]): Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Time difference distribution for camera1 -> camera2 transitions
        transitions = data[data["appear_camera1_then_camera2"] == 1]
        time_diffs = transitions.apply(
            lambda x: (
                x["first_seen_camera2"] - x["last_seen_camera1"]
            ).total_seconds(),
            axis=1,
        )

        sns.histplot(time_diffs, ax=ax1)
        ax1.set_title("Time Between Camera Appearances")
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Count")

        # Movement flow diagram
        labels = ["Camera 1 Only", "Both Cameras", "Camera 2 Only"]
        sizes = [
            len(
                data[
                    (data["appear_in_camera_1"] == 1)
                    & (data["appear_in_camera_2"] == 0)
                ]
            ),
            len(
                data[
                    (data["appear_in_camera_1"] == 1)
                    & (data["appear_in_camera_2"] == 1)
                ]
            ),
            len(
                data[
                    (data["appear_in_camera_1"] == 0)
                    & (data["appear_in_camera_2"] == 1)
                ]
            ),
        ]

        ax2.pie(sizes, labels=labels, autopct="%1.1f%%")
        ax2.set_title("Movement Distribution")

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path)
            plt.close()

    def create_heatmap(
        self,
        data: pd.DataFrame,
        pivot_columns: List[str],
        values_column: str,
        save_path: Optional[str] = None,
    ):
        """
        Create a heatmap visualization.

        Args:
            data (pd.DataFrame): Input data
            pivot_columns (List[str]): Columns to pivot
            values_column (str): Column for values
            save_path (Optional[str]): Path to save the plot
        """
        pivot_table = pd.pivot_table(
            data,
            values=values_column,
            index=pivot_columns[0],
            columns=pivot_columns[1],
            aggfunc="count",
        )

        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt="g", cmap="YlOrRd")

        plt.title(f'Heatmap of {values_column} by {" vs ".join(pivot_columns)}')

        if save_path:
            plt.savefig(self.output_dir / save_path)
            plt.close()

    def plot_time_series(
        self,
        data: pd.DataFrame,
        time_column: str,
        value_column: str,
        groupby_column: Optional[str] = None,
        interval: str = "1H",
        save_path: Optional[str] = None,
    ):
        """
        Create time series visualization.

        Args:
            data (pd.DataFrame): Input data
            time_column (str): Column containing timestamps
            value_column (str): Column containing values to plot
            groupby_column (Optional[str]): Column to group by
            interval (str): Time interval for resampling
            save_path (Optional[str]): Path to save the plot
        """
        # Resample data
        data = data.set_index(time_column)

        if groupby_column:
            grouped = data.groupby(groupby_column)
            plt.figure(figsize=(15, 6))

            for name, group in grouped:
                resampled = group[value_column].resample(interval).count()
                plt.plot(resampled.index, resampled.values, label=name)

            plt.legend()

        else:
            resampled = data[value_column].resample(interval).count()
            plt.figure(figsize=(15, 6))
            plt.plot(resampled.index, resampled.values)

        plt.title(f"Time Series of {value_column}")
        plt.xlabel("Time")
        plt.ylabel("Count")
        plt.xticks(rotation=45)

        if save_path:
            plt.savefig(self.output_dir / save_path)
            plt.close()

    def plot_dwell_time_analysis(
        self, data: pd.DataFrame, save_path: Optional[str] = None
    ):
        """
        Analyze and visualize dwell times.

        Args:
            data (pd.DataFrame): Detection data
            save_path (Optional[str]): Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Dwell time distributions
        sns.boxplot(data=data, y="dwell_time_camera1", ax=ax1)
        ax1.set_title("Dwell Time Distribution - Camera 1")
        ax1.set_ylabel("Dwell Time (seconds)")

        sns.boxplot(data=data, y="dwell_time_camera2", ax=ax2)
        ax2.set_title("Dwell Time Distribution - Camera 2")
        ax2.set_ylabel("Dwell Time (seconds)")

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path)
            plt.close()

    def create_summary_dashboard(
        self, data: pd.DataFrame, save_path: Optional[str] = None
    ):
        """
        Create a comprehensive dashboard of visualizations.

        Args:
            data (pd.DataFrame): Detection data
            save_path (Optional[str]): Path to save the dashboard
        """
        # Create subplot grid
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Traffic Flow Over Time",
                "Age Distribution",
                "Gender Distribution",
                "Camera Appearances",
                "Dwell Time Analysis",
                "Movement Patterns",
            ),
        )

        # Traffic flow
        fig.add_trace(
            go.Histogram(
                x=data[data["appear_in_camera_1"] == 1]["first_seen_camera1"],
                name="Camera 1",
            ),
            row=1,
            col=1,
        )

        # Age distribution
        fig.add_trace(
            go.Bar(
                x=data["age_group"].value_counts().index,
                y=data["age_group"].value_counts().values,
                name="Age Groups",
            ),
            row=1,
            col=2,
        )

        # Gender distribution
        fig.add_trace(
            go.Pie(
                labels=data["gender"].value_counts().index,
                values=data["gender"].value_counts().values,
                name="Gender",
            ),
            row=2,
            col=1,
        )

        # Camera appearances
        camera_data = pd.DataFrame(
            {
                "Camera": ["Camera 1", "Camera 2"],
                "Count": [
                    data["appear_in_camera_1"].sum(),
                    data["appear_in_camera_2"].sum(),
                ],
            }
        )
        fig.add_trace(
            go.Bar(
                x=camera_data["Camera"],
                y=camera_data["Count"],
                name="Camera Appearances",
            ),
            row=2,
            col=2,
        )

        # Dwell time
        fig.add_trace(
            go.Box(y=data["dwell_time_camera1"], name="Camera 1 Dwell Time"),
            row=3,
            col=1,
        )

        # Movement patterns
        movement_data = pd.DataFrame(
            {
                "Pattern": ["Camera 1 Only", "Both Cameras", "Camera 2 Only"],
                "Count": [
                    len(
                        data[
                            (data["appear_in_camera_1"] == 1)
                            & (data["appear_in_camera_2"] == 0)
                        ]
                    ),
                    len(
                        data[
                            (data["appear_in_camera_1"] == 1)
                            & (data["appear_in_camera_2"] == 1)
                        ]
                    ),
                    len(
                        data[
                            (data["appear_in_camera_1"] == 0)
                            & (data["appear_in_camera_2"] == 1)
                        ]
                    ),
                ],
            }
        )
        fig.add_trace(
            go.Pie(
                labels=movement_data["Pattern"],
                values=movement_data["Count"],
                name="Movement Patterns",
            ),
            row=3,
            col=2,
        )

        # Update layout
        fig.update_layout(
            height=1200, showlegend=True, title_text="CCTV Analysis Dashboard"
        )

        if save_path:
            fig.write_html(self.output_dir / save_path)

        return fig


# Example usage
if __name__ == "__main__":
    viz_manager = VisualizationManager()

    # Create sample data
    np.random.seed(42)
    n_samples = 100

    # Generate timestamps for a 24-hour period
    base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    camera1_times = [
        base_time + timedelta(minutes=np.random.randint(0, 1440))
        for _ in range(n_samples)
    ]
    camera2_times = [
        t + timedelta(minutes=np.random.randint(5, 30)) for t in camera1_times
    ]

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

    # Calculate dwell times in seconds
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

    # Create various visualizations

    # 1. Traffic flow analysis
    viz_manager.plot_traffic_flow(data=data, save_path="traffic_flow.png")

    # 2. Interactive traffic flow
    viz_manager.plot_traffic_flow(
        data=data, save_path="traffic_flow_interactive.html", interactive=True
    )

    # 3. Demographics analysis
    viz_manager.plot_demographics(data=data, save_path="demographics.png")

    # 4. Movement patterns
    viz_manager.plot_movement_patterns(data=data, save_path="movement_patterns.png")

    # 5. Dwell time analysis
    viz_manager.plot_dwell_time_analysis(data=data, save_path="dwell_time.png")

    # 6. Create heatmap of age groups vs gender
    viz_manager.create_heatmap(
        data=data,
        pivot_columns=["age_group", "gender"],
        values_column="individual_id",
        save_path="demographic_heatmap.png",
    )

    # 7. Time series analysis of appearances
    viz_manager.plot_time_series(
        data=data,
        time_column="first_seen_camera1",
        value_column="individual_id",
        groupby_column="gender",
        interval="1H",
        save_path="time_series.png",
    )

    # 8. Create comprehensive dashboard
    viz_manager.create_summary_dashboard(data=data, save_path="dashboard.html")

    print("All visualizations have been created successfully!")
    print(f"Visualizations saved in: {viz_manager.output_dir}")
