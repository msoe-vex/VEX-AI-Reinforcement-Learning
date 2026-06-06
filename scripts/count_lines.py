#!/usr/bin/env python3

from __future__ import annotations

import fnmatch
import os
from pathlib import Path


WHITELIST_GLOBS = ["*.py", "*.sh", "*.bat"]
EXTRA_IGNORE_GLOBS = [
	".git",
	".git/**",
	".vscode/**",
	"venv/**",
]


def is_whitelisted(path: Path) -> bool:
	return any(fnmatch.fnmatch(path.name, pattern) for pattern in WHITELIST_GLOBS)


def load_gitignore_globs(root: Path) -> list[str]:
	gitignore_path = root / ".gitignore"
	if not gitignore_path.exists():
		return []

	globs: list[str] = []
	with gitignore_path.open("r", encoding="utf-8", errors="replace") as file_handle:
		for raw_line in file_handle:
			line = raw_line.strip()
			if not line or line.startswith("#"):
				continue
			globs.append(line)
	return globs


def matches_ignore_glob(relative_path: str, pattern: str) -> bool:
	negated = pattern.startswith("!")
	if negated:
		pattern = pattern[1:]

	if pattern in {"/", ""}:
		return False

	normalized_path = relative_path.replace(os.sep, "/")
	pattern = pattern.replace(os.sep, "/")

	if pattern.endswith("/"):
		directory_pattern = pattern.rstrip("/")
		if "/" in directory_pattern:
			return normalized_path == directory_pattern or normalized_path.startswith(directory_pattern + "/")
		return any(part == directory_pattern for part in normalized_path.split("/"))

	if "/" in pattern:
		return fnmatch.fnmatch(normalized_path, pattern)

	return any(fnmatch.fnmatch(part, pattern) for part in normalized_path.split("/"))


def is_ignored(relative_path: str, ignore_globs: list[str]) -> bool:
	ignored = False
	for pattern in ignore_globs:
		if matches_ignore_glob(relative_path, pattern):
			ignored = not pattern.startswith("!")
	return ignored


def count_lines(path: Path) -> int:
	with path.open("r", encoding="utf-8", errors="replace") as file_handle:
		return sum(1 for _ in file_handle)


def main() -> int:
	root = Path.cwd()
	ignore_globs = list(EXTRA_IGNORE_GLOBS) + load_gitignore_globs(root)
	counted_paths: list[Path] = []

	for current_root, directory_names, file_names in os.walk(root):
		current_dir = Path(current_root)
		directory_names[:] = [
			name
			for name in directory_names
			if not is_ignored(str((current_dir / name).relative_to(root)), ignore_globs)
		]

		for file_name in file_names:
			path = current_dir / file_name
			relative_path = str(path.relative_to(root))
			if ".git" in path.parts:
				continue
			if not is_whitelisted(path):
				continue
			if is_ignored(relative_path, ignore_globs):
				continue
			counted_paths.append(path)

	file_stats: list[tuple[int, Path]] = []
	for path in counted_paths:
		file_stats.append((count_lines(path), path))

	total_lines = 0
	for line_count, path in sorted(file_stats, key=lambda item: (-item[0], str(item[1].relative_to(root)))):
		total_lines += line_count
		print(f"{line_count:>8}  {path.relative_to(root)}")

	print("-" * 40)
	print(f"{total_lines:>8}  total")
	print(f"{len(counted_paths):>8}  files counted")
	print(f"{len(ignore_globs):>8}  ignore globs loaded")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
