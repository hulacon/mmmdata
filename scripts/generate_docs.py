#!/usr/bin/env python3
"""
Generate Jekyll-compatible markdown documentation from Python utility modules.

This script parses Python source files, extracts function docstrings,
and generates .md files suitable for Jekyll documentation with the just-the-docs theme.
"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse


class FunctionDocExtractor:
    """Extract documentation from Python functions."""

    def __init__(self, source_file: Path):
        self.source_file = source_file
        self.module_name = source_file.stem
        with open(source_file, 'r') as f:
            self.source = f.read()
        self.tree = ast.parse(self.source)

    def extract_functions(self) -> List[Dict[str, Any]]:
        """Extract all function definitions with their docstrings."""
        functions = []

        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                # Skip private functions (starting with _)
                if node.name.startswith('_'):
                    continue

                func_info = {
                    'name': node.name,
                    'signature': self._get_signature(node),
                    'docstring': ast.get_docstring(node) or '',
                    'line_number': node.lineno
                }

                # Parse the docstring
                parsed_doc = self._parse_docstring(func_info['docstring'])
                func_info.update(parsed_doc)

                functions.append(func_info)

        return functions

    def _get_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature as a string."""
        args = []

        # Regular arguments
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)

        # Default values
        defaults = node.args.defaults
        if defaults:
            num_defaults = len(defaults)
            for i, default in enumerate(defaults):
                idx = len(args) - num_defaults + i
                args[idx] += f" = {ast.unparse(default)}"

        # Return annotation
        return_annotation = ""
        if node.returns:
            return_annotation = f" -> {ast.unparse(node.returns)}"

        return f"{node.name}({', '.join(args)}){return_annotation}"

    def _parse_docstring(self, docstring: str) -> Dict[str, Any]:
        """Parse NumPy-style docstring into structured data."""
        if not docstring:
            return {
                'description': '',
                'parameters': [],
                'returns': '',
                'examples': ''
            }

        lines = docstring.split('\n')

        # Extract description (everything before first section)
        description_lines = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line in ['Parameters', 'Returns', 'Examples', 'Raises', 'Notes']:
                break
            description_lines.append(lines[i])
            i += 1

        description = '\n'.join(description_lines).strip()

        # Extract sections
        sections = self._extract_sections(docstring)

        return {
            'description': description,
            'parameters': sections.get('Parameters', []),
            'returns': sections.get('Returns', ''),
            'examples': sections.get('Examples', ''),
            'raises': sections.get('Raises', ''),
            'notes': sections.get('Notes', '')
        }

    def _extract_sections(self, docstring: str) -> Dict[str, Any]:
        """Extract sections from NumPy-style docstring."""
        sections = {}
        lines = docstring.split('\n')

        current_section = None
        section_content = []

        for line in lines:
            stripped = line.strip()

            # Check if this is a section header
            if stripped in ['Parameters', 'Returns', 'Examples', 'Raises', 'Notes', 'See Also']:
                # Save previous section
                if current_section:
                    sections[current_section] = self._process_section(
                        current_section, section_content
                    )

                current_section = stripped
                section_content = []
            elif current_section:
                section_content.append(line)

        # Save last section
        if current_section:
            sections[current_section] = self._process_section(
                current_section, section_content
            )

        return sections

    def _process_section(self, section_name: str, content: List[str]) -> Any:
        """Process section content based on section type."""
        if section_name == 'Parameters':
            return self._parse_parameters(content)
        else:
            # For other sections, just return cleaned text
            return '\n'.join(content).strip()

    def _parse_parameters(self, lines: List[str]) -> List[Dict[str, str]]:
        """Parse parameter section into structured format."""
        parameters = []
        current_param = None

        for line in lines:
            # Skip separator lines
            if re.match(r'^-+$', line.strip()):
                continue

            # Check if this is a parameter definition line
            match = re.match(r'^\s*(\w+)\s*:\s*(.+)$', line)
            if match:
                # Save previous parameter
                if current_param:
                    parameters.append(current_param)

                param_name = match.group(1)
                param_type = match.group(2).strip()

                current_param = {
                    'name': param_name,
                    'type': param_type,
                    'description': ''
                }
            elif current_param and line.strip():
                # Continuation of description
                if current_param['description']:
                    current_param['description'] += ' '
                current_param['description'] += line.strip()

        # Save last parameter
        if current_param:
            parameters.append(current_param)

        return parameters


class JekyllDocGenerator:
    """Generate Jekyll-compatible markdown documentation."""

    def __init__(self, output_dir: Path, base_nav_order: int = 50):
        self.output_dir = output_dir
        self.base_nav_order = base_nav_order
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_module_docs(self, source_file: Path, module_nav_order: int) -> List[str]:
        """Generate documentation for a Python module."""
        extractor = FunctionDocExtractor(source_file)
        functions = extractor.extract_functions()

        if not functions:
            print(f"No public functions found in {source_file}")
            return []

        module_name = source_file.stem
        generated_files = []

        # Generate individual function pages
        for i, func in enumerate(functions):
            md_file = self.output_dir / f"{module_name}_{func['name']}.md"
            self._write_function_page(func, module_name, md_file, module_nav_order, i)
            generated_files.append(md_file.name)

        return generated_files

    def _write_function_page(
        self,
        func: Dict[str, Any],
        module_name: str,
        output_file: Path,
        parent_nav_order: int,
        func_index: int
    ):
        """Write a single function's documentation page."""
        nav_order = parent_nav_order + (func_index + 1) * 0.01

        with open(output_file, 'w') as f:
            # Jekyll front matter
            f.write("---\n")
            f.write(f"title: {func['name']}\n")
            f.write(f"parent: {self._format_module_name(module_name)}\n")
            f.write(f"grand_parent: Code Documentation\n")
            f.write(f"nav_order: {nav_order:.2f}\n")
            f.write("---\n\n")

            # Function header
            f.write(f"# `{func['name']}`\n\n")

            # Description
            if func['description']:
                f.write(f"{func['description']}\n\n")

            # Signature
            f.write("## Signature\n\n")
            f.write("```python\n")
            f.write(func['signature'])
            f.write("\n```\n\n")

            # Parameters
            if func['parameters']:
                f.write("## Parameters\n\n")
                for param in func['parameters']:
                    f.write(f"**`{param['name']}`** : `{param['type']}`\n")
                    if param['description']:
                        f.write(f"  \n{param['description']}\n\n")

            # Returns
            if func['returns']:
                f.write("## Returns\n\n")
                f.write(f"{func['returns']}\n\n")

            # Examples
            if func['examples']:
                f.write("## Examples\n\n")
                f.write("```python\n")
                f.write(func['examples'])
                f.write("\n```\n\n")

            # Notes
            if func.get('notes'):
                f.write("## Notes\n\n")
                f.write(f"{func['notes']}\n\n")

            # Source reference
            f.write("## Source\n\n")
            f.write(f"Defined in `{module_name}.py` at line {func['line_number']}\n")

    def generate_module_index(self, module_name: str, function_names: List[str], nav_order: int) -> Path:
        """Generate an index page for a module."""
        index_file = self.output_dir / f"{module_name}_index.md"

        with open(index_file, 'w') as f:
            # Jekyll front matter
            f.write("---\n")
            f.write(f"title: {self._format_module_name(module_name)}\n")
            f.write(f"parent: Code Documentation\n")
            f.write(f"nav_order: {nav_order}\n")
            f.write("has_children: true\n")
            f.write("---\n\n")

            # Module header
            f.write(f"# {self._format_module_name(module_name)}\n\n")
            f.write(f"Documentation for functions in the `{module_name}` module.\n\n")

            # Function list
            f.write("## Functions\n\n")
            for func_name in function_names:
                f.write(f"- [{func_name}]({module_name}_{func_name})\n")

        return index_file

    def generate_main_index(self, modules: List[Dict[str, Any]]) -> Path:
        """Generate the main Code Documentation index page."""
        index_file = self.output_dir / "code_index.md"

        with open(index_file, 'w') as f:
            # Jekyll front matter
            f.write("---\n")
            f.write("title: Code Documentation\n")
            f.write(f"nav_order: {self.base_nav_order}\n")
            f.write("has_children: true\n")
            f.write("---\n\n")

            # Header
            f.write("# Code Documentation\n\n")
            f.write("API documentation for Python utility functions in the MMMData project.\n\n")

            # Module list
            f.write("## Available Modules\n\n")
            for module in modules:
                f.write(f"### [{self._format_module_name(module['name'])}]({module['name']}_index)\n\n")
                if module['functions']:
                    for func in module['functions']:
                        f.write(f"- [`{func}`]({module['name']}_{func})\n")
                    f.write("\n")

        return index_file

    def _format_module_name(self, module_name: str) -> str:
        """Format module name for display."""
        return module_name.replace('_', ' ').title()


def main():
    parser = argparse.ArgumentParser(
        description='Generate Jekyll documentation from Python utility modules'
    )
    parser.add_argument(
        'source_files',
        nargs='+',
        type=Path,
        help='Python source files to document'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('docs/doc/code'),
        help='Output directory for generated markdown files (default: docs/doc/code)'
    )
    parser.add_argument(
        '--nav-order',
        type=int,
        default=50,
        help='Base navigation order for the Code Documentation section (default: 50)'
    )

    args = parser.parse_args()

    # Initialize generator
    generator = JekyllDocGenerator(args.output_dir, args.nav_order)

    # Process each module
    modules = []
    for i, source_file in enumerate(args.source_files):
        if not source_file.exists():
            print(f"Warning: {source_file} not found, skipping")
            continue

        print(f"Processing {source_file}...")
        module_nav_order = args.nav_order + i + 1

        # Extract functions
        extractor = FunctionDocExtractor(source_file)
        functions = extractor.extract_functions()
        function_names = [f['name'] for f in functions]

        if not function_names:
            continue

        # Generate function pages
        generator.generate_module_docs(source_file, module_nav_order)

        # Generate module index
        generator.generate_module_index(source_file.stem, function_names, module_nav_order)

        modules.append({
            'name': source_file.stem,
            'functions': function_names
        })

        print(f"  Generated docs for {len(function_names)} functions")

    # Generate main index
    if modules:
        main_index = generator.generate_main_index(modules)
        print(f"\nGenerated main index: {main_index}")
        print(f"Total modules documented: {len(modules)}")
    else:
        print("No modules to document")


if __name__ == '__main__':
    main()
