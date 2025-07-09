#!/usr/bin/env python3

"""
This script parses docs/ops/*.txt and creates the ops.md, which is a table documenting supported operations on various ggml backends.
"""
import os
import re
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


class DocsGenerator:
    def __init__(self, ggml_root: str):
        self.ggml_root = Path(ggml_root)
        self.ops_dir = self.ggml_root / "docs" / "ops"
        self.backend_support: Dict[str, Dict[str, List[bool]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.all_operations: Set[str] = set()
        self.all_backends: Set[str] = set()

    def parse_support_files(self) -> None:
        if not self.ops_dir.exists():
            print(f"Warning: ops directory not found: {self.ops_dir}")
            return

        print(f"Parsing support files from {self.ops_dir}...")

        for backend_dir in self.ops_dir.iterdir():
            if not backend_dir.is_dir():
                continue

            backend_name = backend_dir.name
            self.all_backends.add(backend_name)

            print(f"  Processing backend: {backend_name}")

            for support_file in backend_dir.glob("*.txt"):
                print(f"    Reading: {support_file.name}")
                self._parse_support_file(support_file, backend_name)

    def _parse_support_file(self, file_path: Path, backend_name: str) -> None:
        try:
            with open(file_path, "r") as f:
                content = f.read()

            for line in content.split("\n"):
                line = line.strip()

                if line.startswith("supported,"):
                    parts = line.split(",")
                    if len(parts) >= 3:
                        operation = parts[1].strip()
                        supported_str = parts[2].strip()

                        if not operation or operation in [
                            "CONTEXT_ERROR",
                            "BUILD_ERROR",
                        ]:
                            continue

                        is_supported = supported_str.lower() == "yes"

                        self.backend_support[backend_name][operation].append(
                            is_supported
                        )
                        self.all_operations.add(operation)

        except Exception as e:
            print(f"    Error parsing {file_path}: {e}")

    def get_backend_support_status(self, backend: str, operation: str) -> str:
        support_list = self.backend_support[backend].get(operation, [])

        if not support_list:
            return "unsupported"

        all_supported = all(support_list)
        any_supported = any(support_list)

        if all_supported:
            return "supported"
        elif any_supported:
            return "partially supported"
        else:
            return "unsupported"

    def get_support_status(self, operation: str) -> str:
        if operation not in self.all_operations:
            return "unsupported"

        support_count = 0
        total_backends = len(self.all_backends)

        for backend in self.all_backends:
            if self.backend_support[backend].get(operation, False):
                support_count += 1

        if support_count == 0:
            return "unsupported"
        elif support_count == total_backends:
            return "supported"
        else:
            return "partially supported"

    def get_support_symbol(self, status: str) -> str:
        symbols = {"supported": "✅", "partially supported": "🟡", "unsupported": "❌"}
        return symbols.get(status, "❓")

    def generate_markdown(self) -> str:
        lines = []

        lines.append("# GGML Operations")
        lines.append("")
        lines.append("List of GGML operations and backend support status.")
        lines.append("")
        lines.append("Legend:")
        lines.append("- ✅ Fully supported by this backend")
        lines.append("- 🟡 Partially supported by this backend")
        lines.append("- ❌ Not supported by this backend")
        lines.append("")

        backends = sorted(self.all_backends)
        header = "| Operation |"
        for backend in backends:
            header += f" {backend} |"

        separator = "|-----------|"
        for _ in backends:
            separator += "------|"

        lines.append(header)
        lines.append(separator)

        sorted_operations = sorted(self.all_operations)

        for operation in sorted_operations:
            row = f"| {operation:>32} |"

            for backend in backends:
                status = self.get_backend_support_status(backend, operation)
                if status == "supported":
                    symbol = "✅"
                elif status == "partially supported":
                    symbol = "🟡"
                else:
                    symbol = "❌"
                row += f" {symbol} |"

            lines.append(row)

        lines.append("")

        return "\n".join(lines)

    def run(self) -> None:
        print("Parsing GGML operation support files...")
        self.parse_support_files()

        if not self.all_operations:
            print(
                "No operations found. Make sure to run test-backend-ops support > docs/ops/BACKEND/file.txt first."
            )
            return

        print(
            f"Found {len(self.all_operations)} operations across {len(self.all_backends)} backends"
        )

        print("Generating markdown...")
        markdown_content = self.generate_markdown()

        docs_dir = self.ggml_root / "docs"
        docs_dir.mkdir(exist_ok=True)

        ops_file = docs_dir / "ops.md"
        with open(ops_file, "w") as f:
            f.write(markdown_content)

        print(f"Generated: {ops_file}")
        print(f"Operations: {len(self.all_operations)}")
        print(f"Backends: {len(self.all_backends)}")


def main():
    generator = DocsGenerator(".")
    generator.run()


if __name__ == "__main__":
    main()
