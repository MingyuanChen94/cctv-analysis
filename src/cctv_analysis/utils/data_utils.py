import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .logging_utils import setup_logging

logger = setup_logging(__name__)


class DataManager:
    """Manages data processing, storage, and retrieval operations."""

    def __init__(self, base_dir: str = "data"):
        """
        Initialize data manager.

        Args:
            base_dir (str): Base directory for data storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def save_detections_csv(
        self,
        detections: List[Dict],
        matches: List[Dict],
        camera1_file: str,
        camera2_file: str,
        output_path: str,
    ) -> pd.DataFrame:
        """
        Save detection and tracking results to CSV.

        Args:
            detections (List[Dict]): List of all detections
            matches (List[Dict]): List of matches between cameras
            camera1_file (str): Name of camera 1 video file
            camera2_file (str): Name of camera 2 video file
            output_path (str): Path to save CSV file

        Returns:
            pd.DataFrame: Processed data as DataFrame
        """
        try:
            # Process detections into DataFrame format
            processed_data = self._process_detections(
                detections, matches, camera1_file, camera2_file
            )

            # Save to CSV
            output_path = Path(output_path)
            output_path.parent.mkdir(exist_ok=True)
            processed_data.to_csv(output_path, index=False)

            logger.info(f"Saved detections to {output_path}")
            return processed_data

        except Exception as e:
            logger.error(f"Error saving detections: {str(e)}")
            raise

    def _process_detections(
        self,
        detections: List[Dict],
        matches: List[Dict],
        camera1_file: str,
        camera2_file: str,
    ) -> pd.DataFrame:
        """Process raw detections into structured DataFrame."""

        # Helper function for age grouping
        def get_age_group(age: float) -> str:
            if age is None:
                return "Unknown"
            elif age <= 18:
                return "0-18"
            elif age <= 35:
                return "19-35"
            elif age <= 50:
                return "36-50"
            else:
                return "51+"

        # Create records for each unique individual
        individuals = {}

        for detection in detections:
            track_id = detection["track_id"]
            camera_id = detection["camera_id"]

            if track_id not in individuals:
                individuals[track_id] = {
                    "individual_id": track_id,
                    "age_group": get_age_group(detection.get("age")),
                    "gender": detection.get("gender", "Unknown"),
                    "appear_in_camera_1": 0,
                    "appear_in_camera_2": 0,
                    "appear_camera1_then_camera2": 0,
                    "first_seen_camera1": None,
                    "last_seen_camera1": None,
                    "first_seen_camera2": None,
                    "last_seen_camera2": None,
                    "total_appearances_camera1": 0,
                    "total_appearances_camera2": 0,
                    "camera_1_video_file": camera1_file,
                    "camera_2_video_file": camera2_file,
                }

            # Update appearance information
            if camera_id == 1:
                individuals[track_id]["appear_in_camera_1"] = 1
                individuals[track_id]["total_appearances_camera1"] += 1

                timestamp = detection["timestamp"]
                if (
                    individuals[track_id]["first_seen_camera1"] is None
                    or timestamp < individuals[track_id]["first_seen_camera1"]
                ):
                    individuals[track_id]["first_seen_camera1"] = timestamp

                if (
                    individuals[track_id]["last_seen_camera1"] is None
                    or timestamp > individuals[track_id]["last_seen_camera1"]
                ):
                    individuals[track_id]["last_seen_camera1"] = timestamp

            elif camera_id == 2:
                individuals[track_id]["appear_in_camera2"] = 1
                individuals[track_id]["total_appearances_camera2"] += 1

                timestamp = detection["timestamp"]
                if (
                    individuals[track_id]["first_seen_camera2"] is None
                    or timestamp < individuals[track_id]["first_seen_camera2"]
                ):
                    individuals[track_id]["first_seen_camera2"] = timestamp

                if (
                    individuals[track_id]["last_seen_camera2"] is None
                    or timestamp > individuals[track_id]["last_seen_camera2"]
                ):
                    individuals[track_id]["last_seen_camera2"] = timestamp

        # Process matches to identify camera1 -> camera2 sequences
        for match in matches:
            track_id = match["track_id"]
            if track_id in individuals:
                individuals[track_id]["appear_camera1_then_camera2"] = 1

        # Convert to DataFrame
        df = pd.DataFrame(list(individuals.values()))

        # Calculate additional metrics
        df["dwell_time_camera1"] = df.apply(
            lambda x: x["last_seen_camera1"] - x["first_seen_camera1"]
            if x["appear_in_camera_1"] == 1
            else None,
            axis=1,
        )

        df["dwell_time_camera2"] = df.apply(
            lambda x: x["last_seen_camera2"] - x["first_seen_camera2"]
            if x["appear_in_camera2"] == 1
            else None,
            axis=1,
        )

        # Convert timestamps to datetime if needed
        timestamp_columns = [
            "first_seen_camera1",
            "last_seen_camera1",
            "first_seen_camera2",
            "last_seen_camera2",
        ]

        for col in timestamp_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], unit="ms")

        return df

    def load_existing_data(self, file_path: str) -> pd.DataFrame:
        """
        Load existing data from CSV file.

        Args:
            file_path (str): Path to CSV file

        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise

    def merge_detection_results(
        self, existing_data: pd.DataFrame, new_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge new detection results with existing data.

        Args:
            existing_data (pd.DataFrame): Existing detection data
            new_data (pd.DataFrame): New detection data

        Returns:
            pd.DataFrame: Merged data
        """
        try:
            # Concatenate dataframes
            merged = pd.concat([existing_data, new_data], ignore_index=True)

            # Remove duplicates based on individual_id
            merged = merged.drop_duplicates(subset=["individual_id"], keep="last")

            return merged

        except Exception as e:
            logger.error(f"Error merging detection results: {str(e)}")
            raise

    def export_json(self, data: pd.DataFrame, output_path: str):
        """
        Export data to JSON format.

        Args:
            data (pd.DataFrame): Data to export
            output_path (str): Output file path
        """
        try:
            # Convert DataFrame to dictionary
            data_dict = data.to_dict(orient="records")

            # Save to JSON
            output_path = Path(output_path)
            output_path.parent.mkdir(exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(data_dict, f, indent=4, default=str)

            logger.info(f"Exported data to {output_path}")

        except Exception as e:
            logger.error(f"Error exporting to JSON: {str(e)}")
            raise

    def get_statistics(self, data: pd.DataFrame) -> Dict:
        """
        Calculate various statistics from the data.

        Args:
            data (pd.DataFrame): Input data

        Returns:
            Dict: Calculated statistics
        """
        stats = {
            "total_individuals": len(data),
            "camera_appearances": {
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
                "camera1_then_camera2": len(
                    data[data["appear_camera1_then_camera2"] == 1]
                ),
            },
            "demographics": {
                "age_groups": data["age_group"].value_counts().to_dict(),
                "gender": data["gender"].value_counts().to_dict(),
            },
            "dwell_time": {
                "camera1": {
                    "mean": data["dwell_time_camera1"].mean(),
                    "median": data["dwell_time_camera1"].median(),
                    "std": data["dwell_time_camera1"].std(),
                },
                "camera2": {
                    "mean": data["dwell_time_camera2"].mean(),
                    "median": data["dwell_time_camera2"].median(),
                    "std": data["dwell_time_camera2"].std(),
                },
            },
        }

        return stats


# Example usage
if __name__ == "__main__":
    data_manager = DataManager()

    # Example detections and matches
    detections = [
        {"track_id": 1, "camera_id": 1, "timestamp": 1000, "age": 25, "gender": "Male"},
        {"track_id": 1, "camera_id": 2, "timestamp": 2000, "age": 25, "gender": "Male"},
        {
            "track_id": 2,
            "camera_id": 1,
            "timestamp": 1500,
            "age": 35,
            "gender": "Female",
        },
    ]

    matches = [{"track_id": 1, "camera1_time": 1000, "camera2_time": 2000}]

    # Save detections
    df = data_manager.save_detections_csv(
        detections=detections,
        matches=matches,
        camera1_file="camera1.mp4",
        camera2_file="camera2.mp4",
        output_path="data/detections.csv",
    )

    # Calculate statistics
    stats = data_manager.get_statistics(df)
    print("\nStatistics:", json.dumps(stats, indent=2))
