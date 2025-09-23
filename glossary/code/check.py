"""
This module provides functionality to list all .csv files in a given directory and its subdirectories,
and to check the content of these .csv files for specific conditions.

Functions:
    list_csv_files(directory): Returns a list of paths to .csv files in the specified directory and its subdirectories.
    check_first_line(file_path): Checks if the first line of the CSV file is 'source,target'.
    check_content(file_path): Checks for various content issues in the CSV file.
"""

import glob
import os
from typing import List
import pandas as pd
from logger_config import logger


def list_csv_files(directory: str) -> List[str]:
    """
    Returns a list of paths to .csv files in the specified directory and its subdirectories.

    Args:
        directory (str): The directory to search for .csv files.

    Returns:
        list: A list of paths to .csv files.
    """
    # Use glob to find all .csv files in the directory and subdirectories
    csv_files = glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True)
    return csv_files


def check_first_line(file_path: str) -> bool:
    """
    Checks if the CSV file is encoded in utf-8 and if the first line is 'source,target'.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        bool: True if the file is utf-8 encoded and the first line is 'source,target', False otherwise.
    """
    try:
        # print(f"Checking {file_path}")
        with open(file_path, "r", encoding="utf-8") as file:
            first_line = file.readline().strip()
            # Remove UTF-8 BOM if it exists
            if first_line.startswith("\ufeff"):
                # print("Removing UTF-8 BOM")
                first_line = first_line[1:]
            # compare the first line to 'source,target'
            if first_line != "source,target":
                logger.error(f"First line {repr(first_line)} is not 'source,target'")
                return False
            return True
    except UnicodeDecodeError:
        return False


def check_content(file_path: str) -> bool:
    """
    Checks for various content issues in the CSV file, including:
    - Duplicate entries in the 'source' column
    - Empty lines
    - Empty or space-only values in 'source' or 'target' columns
    - Leading or trailing spaces in 'source' or 'target' columns
    - Non-breaking space characters in 'source' or 'target' columns
    - Ensures the DataFrame has only 'source' and 'target' columns

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        bool: True if no issues are found, False otherwise.
    """
    try:
        df = pd.read_csv(
            file_path,
            encoding="utf-8",
            keep_default_na=False,
            skip_blank_lines=False,
        )

        # Print the number of rows in the DataFrame
        if len(df) < 100:
            logger.warning(f"{file_path} has {len(df)} rows. (Warning: less than 100 rows)")
        else:
            logger.info(f"{file_path} has {len(df)} rows.")

        # Print the entire DataFrame content for debugging
        # print(f"DataFrame content of {file_path}:")
        # print(df)

        if list(df.columns) != ["source", "target"]:
            logger.error(f"{file_path} does not have exactly 'source' and 'target' columns.")
            return False

        issues = {
            "duplicates": df[df.duplicated(subset=["source"], keep=False)],
            "empty_lines": df[df.isnull().all(axis=1)],
            "empty_or_space_source": df[df["source"].str.strip() == ""],
            "empty_or_space_target": df[df["target"].str.strip() == ""],
            "leading_trailing_space_source": df[df["source"] != df["source"].str.strip()],
            "leading_trailing_space_target": df[df["target"] != df["target"].str.strip()],
            "non_breaking_space_source": df[df["source"].str.contains("\u00a0")],
            "non_breaking_space_target": df[df["target"].str.contains("\u00a0")],
            "double_quote_source": df[df["source"].str.contains('"')],
            "double_quote_target": df[df["target"].str.contains('"')],
        }

        for issue, data in issues.items():
            if not data.empty:
                logger.error(f"{file_path}: {issue.replace('_', ' ')} found")
                data.index = data.index + 2  # Adjust row numbers by adding 2
                logger.error(data)

        return all(data.empty for data in issues.values())
    except Exception as e:
        logger.error(f"{file_path}: Error reading {e}")
        return False


def main() -> None:
    # Example usage
    data_directory = "./data"
    csv_files = list_csv_files(data_directory)
    for file in csv_files:
        # check if the file ends with index.csv
        if file.endswith("index.csv"):
            logger.info(f"{file} is an index file, skipping.")
            continue
        if check_first_line(file):
            logger.info(f"{file} check_first_line passed.")
        else:
            logger.error(f"{file} check_first_line failed.")

        if check_content(file):
            logger.info(f"{file} check_content passed.")
        else:
            logger.error(f"{file} check_content failed.")


if __name__ == "__main__":
    main()
