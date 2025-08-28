import argparse
import json
import os
import shutil
from typing import Optional


def is_non_empty_list_json(file_path: str) -> Optional[bool]:
	"""Return True if JSON is a list and not empty, False if list is empty, None if file missing.

	If the JSON exists but is not a list (or malformed), treat it as non-empty (returns True),
	so the caller can choose to delete to be safe.
	"""
	if not os.path.isfile(file_path):
		return None
	try:
		with open(file_path, "r", encoding="utf-8") as f:
			data = json.load(f)
			if isinstance(data, list):
				return len(data) > 0
			# Not a list → treat as non-empty
			return True
	except Exception:
		# Malformed or unreadable → treat as non-empty
		return True


def process_parent_directory(parent_dir: str) -> None:
	"""Scan each immediate subdirectory of parent_dir and delete ones with non-empty neurons_not_checked.json."""
	if not os.path.isdir(parent_dir):
		raise NotADirectoryError(f"Provided path is not a directory: {parent_dir}")

	entries = [
		os.path.join(parent_dir, name)
		for name in os.listdir(parent_dir)
		if os.path.isdir(os.path.join(parent_dir, name))
	]

	for experiment_dir in entries:
		marker_path = os.path.join(experiment_dir, "neurons_not_checked.json")
		status = is_non_empty_list_json(marker_path)

		if status is None:
			# No marker file: skip
			continue

		if status:
			print(f"Deleting: {experiment_dir} (neurons_not_checked.json is non-empty or invalid)")
			shutil.rmtree(experiment_dir, ignore_errors=False)
		else:
			print(f"Keeping:  {experiment_dir} (neurons_not_checked.json is an empty list)")


def main() -> None:
	parser = argparse.ArgumentParser(description="Delete experiment folders whose neurons_not_checked.json is not an empty list.")
	parser.add_argument(
		"path",
		help="Path containing experiment subfolders to check",
	)
	args = parser.parse_args()
	process_parent_directory(args.path)


if __name__ == "__main__":
	main()


