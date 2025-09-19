"""
Command Line Interface for Oven Compiler

Provides CLI commands for compiling Python to MLIR.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

from .compiler import PythonToMLIRCompiler


def compile_command(args):
    """Handle the compile command."""
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

    print(f"Oven Compiler version {__version__}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="oven",
        description="Oven Compiler - Python to MLIR compilation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  oven compile kernel.py                 # Compile kernel.py to kernel.mlir
  oven compile kernel.py -o output.mlir  # Specify output file
  oven compile kernel.py --debug         # Enable debug output
  oven compile kernel.py --optimize      # Enable optimization
  oven --version                         # Show version
        """,
    )

    parser.add_argument(
        "--version", action="store_true", help="Show version information"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Compile command
    compile_parser = subparsers.add_parser(
        "compile", help="Compile Python file to MLIR"
    )
    compile_parser.add_argument("input_file", help="Input Python file to compile")
    compile_parser.add_argument(
        "-o",
        "--output",
        dest="output_file",
        help="Output MLIR file (default: input_file.mlir)",
    )
    compile_parser.add_argument(
        "--debug", action="store_true", help="Enable debug output"
    )
    compile_parser.add_argument(
        "--optimize", action="store_true", help="Enable optimization passes"
    )
    compile_parser.set_defaults(func=compile_command)

    # Parse arguments
    args = parser.parse_args()

    # Handle version flag
    if args.version:
        version_command(args)
        return

    # Handle commands
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
