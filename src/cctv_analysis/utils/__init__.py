import base64
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdfkit
from jinja2 import Environment, FileSystemLoader

from .logging_utils import setup_logging

logger = setup_logging(__name__)


class ReportGenerator:
    """Generates comprehensive reports from CCTV analysis results."""

    REPORT_TEMPLATES = {
        "html": "report_template.html",
        "pdf": "report_template.html",  # Same template, different output
        "markdown": "report_template.md",
        "json": None,  # No template needed for JSON
    }

    def __init__(self, template_dir: str = "templates", output_dir: str = "reports"):
        """
        Initialize report generator.

        Args:
            template_dir (str): Directory containing report templates
            output_dir (str): Directory for saving generated reports
        """
        self.template_dir = Path(template_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Jinja2 environment
        self.env = Environment(loader=FileSystemLoader(str(self.template_dir)))

        # Add custom filters
        self.env.filters["format_timestamp"] = self.format_timestamp
        self.env.filters["format_number"] = self.format_number
        self.env.filters["format_percentage"] = self.format_percentage

    def generate_report(
        self,
        data: pd.DataFrame,
        metrics: Dict,
        visualizations: Dict[str, str],
        output_format: str = "html",
        include_images: bool = True,
    ) -> str:
        """
        Generate analysis report.

        Args:
            data (pd.DataFrame): Detection and tracking data
            metrics (Dict): Calculated metrics
            visualizations (Dict[str, str]): Paths to visualization files
            output_format (str): Output format ('html', 'pdf', 'markdown', 'json')
            include_images (bool): Whether to include images in report

        Returns:
            str: Path to generated report
        """
        # Prepare report data
        report_data = self._prepare_report_data(data, metrics)

        if include_images:
            report_data["visualizations"] = self._process_visualizations(visualizations)

        # Generate report based on format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = (
            self.output_dir / f"cctv_analysis_report_{timestamp}.{output_format}"
        )

        if output_format == "json":
            return self._generate_json_report(report_data, output_file)
        else:
            return self._generate_templated_report(
                report_data, output_format, output_file
            )

    def _prepare_report_data(self, data: pd.DataFrame, metrics: Dict) -> Dict:
        """Prepare data for report generation."""
        return {
            "timestamp": datetime.now(),
            "summary": {
                "total_individuals": len(data),
                "analysis_period": {
                    "start": data["first_seen_camera1"].min(),
                    "end": data["last_seen_camera2"].max(),
                },
                "camera_coverage": metrics.get("camera_coverage", {}),
                "key_findings": self._generate_key_findings(metrics),
            },
            "metrics": metrics,
            "demographic_analysis": self._prepare_demographic_analysis(data),
            "movement_analysis": self._prepare_movement_analysis(data),
            "temporal_analysis": self._prepare_temporal_analysis(data),
        }

    def _generate_key_findings(self, metrics: Dict) -> List[str]:
        """Generate key findings from metrics."""
        findings = []

        # Traffic patterns
        if "temporal_analysis" in metrics:
            peak_hours = metrics["temporal_analysis"].get("peak_hours", {})
            if peak_hours:
                findings.append(
                    f"Peak activity observed during hours: {', '.join(map(str, peak_hours))}"
                )

        # Demographics
        if "demographics" in metrics:
            demo = metrics["demographics"]
            if "age_distribution" in demo:
                most_common_age = max(
                    demo["age_distribution"]["counts"].items(), key=lambda x: x[1]
                )
                findings.append(
                    f"Most common age group: {most_common_age[0]} "
                    f"({most_common_age[1]} individuals)"
                )

        # Movement patterns
        if "movement_patterns" in metrics:
            movement = metrics["movement_patterns"]
            if "transitions" in movement:
                avg_time = movement["transitions"].get("average_transition_time")
                if avg_time:
                    findings.append(
                        f"Average transition time between cameras: "
                        f"{self.format_number(avg_time)} seconds"
                    )

        return findings

    def _prepare_demographic_analysis(self, data: pd.DataFrame) -> Dict:
        """Prepare demographic analysis section."""
        return {
            "age_distribution": data["age_group"].value_counts().to_dict(),
            "gender_distribution": data["gender"].value_counts().to_dict(),
            "cross_tabulation": pd.crosstab(
                data["age_group"], data["gender"]
            ).to_dict(),
        }

    def _prepare_movement_analysis(self, data: pd.DataFrame) -> Dict:
        """Prepare movement analysis section."""
        return {
            "camera_transitions": len(data[data["appear_camera1_then_camera2"] == 1]),
            "single_camera_appearances": {
                "camera1_only": len(
                    data[
                        (data["appear_in_camera_1"] == 1)
                        & (data["appear_in_camera2"] == 0)
                    ]
                ),
                "camera2_only": len(
                    data[
                        (data["appear_in_camera_1"] == 0)
                        & (data["appear_in_camera2"] == 1)
                    ]
                ),
            },
            "dwell_time_statistics": {
                "camera1": {
                    "average": data["dwell_time_camera1"].mean(),
                    "median": data["dwell_time_camera1"].median(),
                    "std": data["dwell_time_camera1"].std(),
                },
                "camera2": {
                    "average": data["dwell_time_camera2"].mean(),
                    "median": data["dwell_time_camera2"].median(),
                    "std": data["dwell_time_camera2"].std(),
                },
            },
        }

    def _prepare_temporal_analysis(self, data: pd.DataFrame) -> Dict:
        """Prepare temporal analysis section."""
        return {
            "hourly_distribution": {
                "camera1": data[data["appear_in_camera1"] == 1]["first_seen_camera1"]
                .dt.hour.value_counts()
                .sort_index()
                .to_dict(),
                "camera2": data[data["appear_in_camera2"] == 1]["first_seen_camera2"]
                .dt.hour.value_counts()
                .sort_index()
                .to_dict(),
            },
            "daily_distribution": {
                "camera1": data[data["appear_in_camera1"] == 1]["first_seen_camera1"]
                .dt.day_name()
                .value_counts()
                .to_dict(),
                "camera2": data[data["appear_in_camera2"] == 1]["first_seen_camera2"]
                .dt.day_name()
                .value_counts()
                .to_dict(),
            },
        }

    def _process_visualizations(self, visualizations: Dict[str, str]) -> Dict[str, str]:
        """Process visualization files for inclusion in report."""
        processed = {}

        for name, path in visualizations.items():
            if Path(path).exists():
                if path.endswith(".html"):
                    # For interactive visualizations, store the file path
                    processed[name] = str(Path(path).absolute())
                else:
                    # For static images, convert to base64
                    with open(path, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode()
                        processed[name] = f"data:image/png;base64,{img_data}"

        return processed

    def _generate_json_report(self, report_data: Dict, output_file: Path) -> str:
        """Generate JSON format report."""
        with open(output_file, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        logger.info(f"Generated JSON report: {output_file}")
        return str(output_file)

    def _generate_templated_report(
        self, report_data: Dict, output_format: str, output_file: Path
    ) -> str:
        """Generate report using template."""
        template = self.env.get_template(self.REPORT_TEMPLATES[output_format])
        content = template.render(**report_data)

        if output_format == "html":
            output_file.write_text(content)

        elif output_format == "pdf":
            # Convert HTML to PDF
            pdfkit.from_string(
                content, str(output_file), options={"enable-local-file-access": None}
            )

        elif output_format == "markdown":
            output_file.write_text(content)

        logger.info(f"Generated {output_format.upper()} report: {output_file}")
        return str(output_file)

    @staticmethod
    def format_timestamp(timestamp) -> str:
        """Format timestamp for display."""
        if pd.isna(timestamp):
            return "N/A"
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def format_number(number: float, decimal_places: int = 2) -> str:
        """Format number for display."""
        if pd.isna(number):
            return "N/A"
        return f"{number:,.{decimal_places}f}"

    @staticmethod
    def format_percentage(value: float, decimal_places: int = 1) -> str:
        """Format percentage for display."""
        if pd.isna(value):
            return "N/A"
        return f"{value:.{decimal_places}f}%"


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 100

    # Generate timestamps
    base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    camera1_times = [
        base_time + pd.Timedelta(minutes=np.random.randint(0, 1440))
        for _ in range(n_samples)
    ]
    camera2_times = [
        t + pd.Timedelta(minutes=np.random.randint(5, 30)) for t in camera1_times
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
                t + pd.Timedelta(minutes=np.random.randint(1, 10))
                for t in camera1_times
            ],
            "first_seen_camera2": camera2_times,
            "last_seen_camera2": [
                t + pd.Timedelta(minutes=np.random.randint(1, 10))
                for t in camera2_times
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

    # Sample metrics
    metrics = {
        "camera_coverage": {"camera1_only": 30, "camera2_only": 20, "both_cameras": 50},
        "temporal_analysis": {"peak_hours": [9, 12, 17]},
        "demographics": {
            "age_distribution": {
                "counts": {"19-35": 40, "36-50": 30, "0-18": 20, "51+": 10}
            }
        },
        "movement_patterns": {"transitions": {"average_transition_time": 15.5}},
    }

    # Sample visualization paths
    visualizations = {
        "traffic_flow": "visualizations/traffic_flow.png",
        "demographics": "visualizations/demographics.png",
        "movement_patterns": "visualizations/movement_patterns.png",
    }

    # Generate reports in different formats
    report_generator = ReportGenerator()

    html_report = report_generator.generate_report(
        data=data, metrics=metrics, visualizations=visualizations, output_format="html"
    )
    print(f"Generated HTML report: {html_report}")

    pdf_report = report_generator.generate_report(
        data=data, metrics=metrics, visualizations=visualizations, output_format="pdf"
    )
    print(f"Generated PDF report: {pdf_report}")

    json_report = report_generator.generate_report(
        data=data, metrics=metrics, visualizations=visualizations, output_format="json"
    )
    print(f"Generated JSON report: {json_report}")
