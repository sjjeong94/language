#!/usr/bin/env python3
"""
Python to MLIR Compiler - Command Line Interface

This is the main entry point for the Python to MLIR compiler.
"""

import argparse
import sys
import os
from pathlib import Path

# Add oven directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from oven.compiler import PythonToMLIRCompiler, CompilationError


def main():
    """Main entry point for the compiler CLI."""
    parser = argparse.ArgumentParser(
        description="Compile Python source code to MLIR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py input.py -o output.mlir
  python main.py input.py --debug
  python main.py input.py -o output.mlir --no-optimize
        """,
    )

    parser.add_argument("input", help="Input Python file to compile")

    parser.add_argument(
        "-o", "--output", help="Output MLIR file (default: <input>.mlir)", default=None
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    parser.add_argument(
        "--no-optimize", action="store_true", help="Disable optimizations"
    )

    parser.add_argument(
        "--version", action="version", version="Python to MLIR Compiler 1.0.0"
    )

    parser.add_argument(
        "--print-ast", action="store_true", help="Print Python AST and exit"
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        return 1

    # Determine output file
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.with_suffix(".mlir"))

    try:
        # Create compiler instance
        compiler = PythonToMLIRCompiler(debug=args.debug, optimize=not args.no_optimize)

        if args.print_ast:
            # Just print the AST and exit
            import ast

            with open(args.input, "r", encoding="utf-8") as f:
                source_code = f.read()
            python_ast = ast.parse(source_code, filename=args.input)
            print(ast.dump(python_ast, indent=2))
            return 0

        # Compile the file
        if args.debug:
            print(f"Compiling {args.input} to {args.output}")

        mlir_code = compiler.compile_file(args.input)

        # Save output
        compiler.save_to_file(mlir_code, args.output)

        if args.debug:
            print("\nCompiler Info:")
            info = compiler.get_compiler_info()
            for key, value in info.items():
                print(f"  {key}: {value}")

        print(f"Successfully compiled {args.input} to {args.output}")

        # Print any warnings
        errors = compiler.get_compilation_errors()
        if errors:
            print("\nWarnings:")
            for error in errors:
                print(f"  {error}")

        return 0

    except CompilationError as e:
        print(f"Compilation failed: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
