"""
Python to MLIR Compiler

Main compiler interface that coordinates the compilation process
from Python source code to MLIR output.
"""

import ast
import sys
import os
from typing import Optional, Dict, Any, List
from .ast_visitor import PythonToMLIRASTVisitor
from .mlir_generator import MLIRGenerator


class PythonToMLIRCompiler:
    """
    Main compiler class that orchestrates the Python to MLIR compilation process.

    This class provides the high-level interface for compiling Python source code
    to MLIR intermediate representation.
    """

    def __init__(self, debug: bool = False, optimize: bool = True):
        """
        Initialize the compiler.

        Args:
            debug: Enable debug output
            optimize: Enable basic optimizations
        """
        self.debug = debug
        self.optimize = optimize
        self.compilation_errors: List[str] = []

    def compile_source(self, source_code: str, filename: str = "<string>") -> str:
        """
        Compile Python source code to MLIR.

        Args:
            source_code: Python source code as string
            filename: Source filename for error reporting

        Returns:
            Generated MLIR code as string

        Raises:
            SyntaxError: If Python source has syntax errors
            CompilationError: If compilation fails
        """
        try:
            # Parse Python source to AST
            if self.debug:
                print(f"Parsing Python source from {filename}")

            python_ast = ast.parse(source_code, filename=filename)

            if self.debug:
                print("Python AST parsed successfully")
                print(ast.dump(python_ast, indent=2))

            # Convert AST to MLIR
            mlir_code = self.compile_ast(python_ast)

            return mlir_code

        except SyntaxError as e:
            error_msg = f"Python syntax error in {filename}: {e}"
            self.compilation_errors.append(error_msg)
            raise SyntaxError(error_msg) from e
        except Exception as e:
            error_msg = f"Compilation error: {e}"
            self.compilation_errors.append(error_msg)
            raise CompilationError(error_msg) from e

    def compile_file(self, input_file: str) -> str:
        """
        Compile a Python file to MLIR.

        Args:
            input_file: Path to Python source file

        Returns:
            Generated MLIR code as string
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        try:
            with open(input_file, "r", encoding="utf-8") as f:
                source_code = f.read()

            return self.compile_source(source_code, input_file)

        except IOError as e:
            error_msg = f"Error reading file {input_file}: {e}"
            self.compilation_errors.append(error_msg)
            raise CompilationError(error_msg) from e

    def compile_ast(self, python_ast: ast.AST) -> str:
        """
        Compile a Python AST to MLIR.

        Args:
            python_ast: Python AST node

        Returns:
            Generated MLIR code as string
        """
        # Create AST visitor
        visitor = PythonToMLIRASTVisitor()

        if self.debug:
            print("Starting AST traversal...")

        # Visit the AST to generate MLIR
        visitor.visit(python_ast)

        # Get generated MLIR code
        mlir_code = visitor.get_mlir_code()

        if self.debug:
            print("MLIR generation completed")
            print(f"Generated {len(mlir_code.split(chr(10)))} lines of MLIR code")

        # Apply optimizations if enabled
        if self.optimize:
            mlir_code = self.apply_basic_optimizations(mlir_code)

        return mlir_code

    def apply_basic_optimizations(self, mlir_code: str) -> str:
        """
        Apply basic optimizations to the generated MLIR code.

        Args:
            mlir_code: Raw MLIR code

        Returns:
            Optimized MLIR code
        """
        if self.debug:
            print("Applying basic optimizations...")

        lines = mlir_code.split("\n")
        optimized_lines = []

        # Remove empty lines and redundant comments
        removed_lines = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not (stripped.startswith("//") and "TODO" in stripped):
                optimized_lines.append(line)
            else:
                removed_lines.append(f"Line {i+1}: '{line}'")

        if self.debug and removed_lines:
            print("Lines removed during optimization:")
            for removed in removed_lines:
                print(f"  REMOVED: {removed}")

        # TODO: Add more sophisticated optimizations:
        # - Constant folding
        # - Dead code elimination
        # - Common subexpression elimination

        optimized_code = "\n".join(optimized_lines)

        if self.debug:
            original_lines = len(lines)
            optimized_lines_count = len(optimized_lines)
            print(
                f"Optimization reduced code from {original_lines} to {optimized_lines_count} lines"
            )

        return optimized_code

    def save_to_file(self, mlir_code: str, output_file: str) -> None:
        """
        Save MLIR code to a file.

        Args:
            mlir_code: Generated MLIR code
            output_file: Output file path
        """
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(mlir_code)

            if self.debug:
                print(f"MLIR code saved to {output_file}")

        except IOError as e:
            error_msg = f"Error writing to file {output_file}: {e}"
            self.compilation_errors.append(error_msg)
            raise CompilationError(error_msg) from e

    def get_compilation_errors(self) -> List[str]:
        """Get list of compilation errors."""
        return self.compilation_errors.copy()

    def clear_errors(self) -> None:
        """Clear compilation error list."""
        self.compilation_errors.clear()

    def get_compiler_info(self) -> Dict[str, Any]:
        """Get compiler configuration and status information."""
        return {
            "debug": self.debug,
            "optimize": self.optimize,
            "error_count": len(self.compilation_errors),
            "supported_python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "mlir_dialects": ["func", "arith", "cf", "memref", "llvm"],
        }


class CompilationError(Exception):
    """Exception raised during compilation process."""

    pass


# Convenience functions for common use cases
def compile_python_string(source: str, debug: bool = False) -> str:
    """
    Compile a Python string to MLIR.

    Args:
        source: Python source code
        debug: Enable debug output

    Returns:
        Generated MLIR code
    """
    compiler = PythonToMLIRCompiler(debug=debug)
    return compiler.compile_source(source)


def compile_python_file(
    input_file: str, output_file: Optional[str] = None, debug: bool = False
) -> str:
    """
    Compile a Python file to MLIR.

    Args:
        input_file: Input Python file path
        output_file: Output MLIR file path (optional)
        debug: Enable debug output

    Returns:
        Generated MLIR code
    """
    compiler = PythonToMLIRCompiler(debug=debug)
    mlir_code = compiler.compile_file(input_file)

    if output_file:
        compiler.save_to_file(mlir_code, output_file)

    return mlir_code
