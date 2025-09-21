"""
Command Line Interface for Oven Language

Provides CLI commands for compiling Python to MLIR.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

from .compiler import PythonToMLIRCompiler


def compile_file_command(args):
    """Handle direct file compilation."""
    input_file = args.input_file
    output_file = args.output_file
    debug = args.debug
    optimize = args.optimize

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    # Determine output file if not specified
    if not output_file:
        input_path = Path(input_file)
        output_file = input_path.with_suffix(".mlir")

    try:
        # Create compiler instance
        compiler = PythonToMLIRCompiler(debug=debug, optimize=optimize)

        # Compile file
        mlir_code = compiler.compile_file(input_file)

        # Write output
        with open(output_file, "w") as f:
            f.write(mlir_code)

        print(f"Successfully compiled {input_file} to {output_file}")

    except Exception as e:
        print(f"Error compiling {input_file}: {e}")
        if debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def version_command(args):
    """Handle the version command."""
    from . import __version__

    print(f"Oven Language version {__version__}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="oven",
        description="Oven Language - Python to MLIR compilation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  oven kernel.py                 # Compile kernel.py to kernel.mlir
  oven kernel.py -o output.mlir  # Specify output file
  oven kernel.py --debug         # Enable debug output
  oven kernel.py --optimize      # Enable optimization
  oven --version                 # Show version
        """,
    )

    parser.add_argument(
        "--version", action="store_true", help="Show version information"
    )

    parser.add_argument("input_file", nargs="?", help="Input Python file to compile")

    parser.add_argument(
        "-o",
        "--output",
        dest="output_file",
        help="Output MLIR file (default: input_file.mlir)",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    parser.add_argument(
        "--optimize", action="store_true", help="Enable optimization passes"
    )

    # Parse arguments
    args = parser.parse_args()

    # Handle version flag
    if args.version:
        version_command(args)
        return

    # Handle file compilation
    if args.input_file:
        compile_file_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
