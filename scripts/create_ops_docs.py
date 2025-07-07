#!/usr/bin/env python3

import os
import re
from pathlib import Path
from typing import Dict, List, Set

GGML_SUPPORTED_BACKENDS = {
            'CPU': '',
            'CUDA': 'src/ggml-cuda/ggml-cuda.cu', 
            'Metal': 'src/ggml-metal/ggml-metal.m',
            'Vulkan': 'src/ggml-vulkan/ggml-vulkan.cpp',
            'OpenCL': 'src/ggml-opencl/ggml-opencl.cpp',
            'CANN': 'src/ggml-cann/ggml-cann.cpp'
}

class DocsGenerator:
    def __init__(self, ggml_root: str):
        self.ggml_root = Path(ggml_root)
        self.operations: List[str] = []
        self.backend_support: Dict[str, Set[str]] = {}
        
    def parse_operations(self) -> None:
        header_file = self.ggml_root / "include" / "ggml.h"
        if not header_file.exists():
            raise FileNotFoundError(f"GGML header not found: {header_file}")
            
        with open(header_file, 'r') as f:
            content = f.read()
            
        # Parse main operations enum
        enum_pattern = r'enum ggml_op \{(.*?)\};'
        enum_match = re.search(enum_pattern, content, re.DOTALL)
        
        if enum_match:
            enum_content = enum_match.group(1)
            lines = enum_content.strip().split('\n')
            
            for line in lines:
                line = line.strip().rstrip(',')
                if not line or line.startswith('//'):
                    continue
                    
                # Extract operation name
                if '=' in line:
                    op_name = line.split('=')[0].strip()
                else:
                    op_name = line.strip()
                
                # Clean up operation name
                op_name = op_name.split('//')[0].strip()
                
                # Skip COUNT entries
                if op_name and not op_name.endswith('_COUNT'):
                    self.operations.append(op_name)
    
    def parse_backend_support(self) -> None:
       
        # CPU always supports everything
        self.backend_support['CPU'] = set(self.operations)
        
        for backend_name, file_path in GGML_SUPPORTED_BACKENDS.items():

            if backend_name == "CPU": continue

            full_path = self.ggml_root / file_path
            self.backend_support[backend_name] = set()
            
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                    
                    # Find operations mentioned in the file
                    for op in self.operations:
                        if op+":" in content:
                            self.backend_support[backend_name].add(op)
                            
                except Exception as e:
                    print(f"Warning: Could not parse {file_path}: {e}")
    
    def generate_markdown(self) -> str:
        lines = []
        
        # Header
        lines.append("# GGML Operations")
        lines.append("")
        lines.append("List of GGML operations and backend support.")
        lines.append("")
        
        # Create table header
        backends = list(GGML_SUPPORTED_BACKENDS.keys())
        header = "| Operation | " + " | ".join(backends) + " |"
        separator = "|" + "|".join(["-" * (len(col) + 2) for col in ["Operation"] + backends]) + "|"
        
        lines.append(header)
        lines.append(separator)
        
        # Add operations
        for op in sorted(self.operations):
            row = f"| {op} |"
            for backend in backends:
                if backend == 'CPU':
                    # CPU always supports everything
                    row += " ✅ |"
                elif op in self.backend_support.get(backend, set()):
                    row += " ✅ |"
                else:
                    row += " ❌ |"
            lines.append(row)
        
        lines.append("")
        
        # Stats
        total_ops = len(self.operations)
        lines.append(f"Total operations: {total_ops}")
        lines.append("")
        
        return "\n".join(lines)
    
    def run(self) -> None:
        """Generate the documentation"""
        print("Parsing GGML operations...")
        self.parse_operations()
        
        print("Checking backend support...")
        self.parse_backend_support()
        
        print("Generating markdown...")
        markdown_content = self.generate_markdown()
        
        # Create docs directory
        docs_dir = self.ggml_root / "docs"
        docs_dir.mkdir(exist_ok=True)
        
        # Write ops.md
        ops_file = docs_dir / "ops.md"
        with open(ops_file, 'w') as f:
            f.write(markdown_content)
        
        print(f"Generated: {ops_file}")
        print(f"Operations: {len(self.operations)}")
        print(f"Backends: {len(self.backend_support)}")

def main():
    generator = DocsGenerator(".")
    generator.run()

if __name__ == "__main__":
    main()
