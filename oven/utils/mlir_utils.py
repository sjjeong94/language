"""
MLIR Utility Functions

This module provides utility functions for MLIR code generation,
including type management, validation, and formatting helpers.
"""

from typing import Dict, List, Optional, Union, Any


class MLIRUtils:
    """
    Utility class for MLIR code generation.

    Provides helper functions for type checking, validation,
    and common MLIR operations.
    """

    # MLIR type mappings
    PYTHON_TO_MLIR_TYPES = {
        "int": "i32",
        "float": "f32",
        "bool": "i1",
        "str": "!llvm.ptr<i8>",
    }

    # MLIR operation mappings
    BINARY_OPS = {
        "add": "arith.addi",
        "sub": "arith.subi",
        "mul": "arith.muli",
        "div": "arith.divsi",
        "mod": "arith.remsi",
        "and": "arith.andi",
        "or": "arith.ori",
        "xor": "arith.xori",
    }

    COMPARE_OPS = {
        "eq": "eq",
        "ne": "ne",
        "lt": "slt",
        "le": "sle",
        "gt": "sgt",
        "ge": "sge",
    }

    @staticmethod
    def python_type_to_mlir(python_type: str) -> str:
        """Convert Python type name to MLIR type."""
        return MLIRUtils.PYTHON_TO_MLIR_TYPES.get(python_type, "i32")

    @staticmethod
    def get_binary_op(op_name: str) -> str:
        """Get MLIR binary operation name."""
        return MLIRUtils.BINARY_OPS.get(op_name, "arith.addi")

    @staticmethod
    def get_compare_op(op_name: str) -> str:
        """Get MLIR comparison operation name."""
        return MLIRUtils.COMPARE_OPS.get(op_name, "eq")

    @staticmethod
    def is_valid_ssa_name(name: str) -> bool:
        """Check if a string is a valid SSA value name."""
        return name.startswith("%") and len(name) > 1

    @staticmethod
    def is_valid_block_label(label: str) -> bool:
        """Check if a string is a valid block label."""
        return label.startswith("^") and len(label) > 1

    @staticmethod
    def sanitize_identifier(name: str) -> str:
        """Sanitize a Python identifier for use in MLIR."""
        # Replace invalid characters with underscores
        sanitized = ""
        for char in name:
            if char.isalnum() or char == "_":
                sanitized += char
            else:
                sanitized += "_"

        # Ensure it starts with a letter or underscore
        if sanitized and not (sanitized[0].isalpha() or sanitized[0] == "_"):
            sanitized = "_" + sanitized

        return sanitized or "_unnamed"

    @staticmethod
    def format_function_signature(
        name: str, arg_types: List[str], return_type: str
    ) -> str:
        """Format a function signature for MLIR."""
        args_str = ", ".join(
            f"%arg{i}: {arg_type}" for i, arg_type in enumerate(arg_types)
        )
        return f"func.func @{name}({args_str}) -> {return_type}"

    @staticmethod
    def format_operation(
        op_name: str,
        operands: List[str],
        result_type: str = "",
        attributes: Dict[str, Any] = None,
    ) -> str:
        """Format a generic MLIR operation."""
        op_str = op_name

        if operands:
            op_str += " " + ", ".join(operands)

        if attributes:
            attr_strs = []
            for key, value in attributes.items():
                if isinstance(value, str):
                    attr_strs.append(f'{key} = "{value}"')
                else:
                    attr_strs.append(f"{key} = {value}")
            if attr_strs:
                op_str += " {" + ", ".join(attr_strs) + "}"

        if result_type:
            op_str += f" : {result_type}"

        return op_str

    @staticmethod
    def escape_string_literal(value: str) -> str:
        """Escape a string literal for MLIR."""
        return (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\t", "\\t")
        )

    @staticmethod
    def get_default_value_for_type(mlir_type: str) -> str:
        """Get default value for an MLIR type."""
        if mlir_type.startswith("i"):
            return "0"
        elif mlir_type.startswith("f"):
            return "0.0"
        elif mlir_type == "i1":
            return "false"
        else:
            return "0"

    @staticmethod
    def validate_mlir_syntax(code: str) -> List[str]:
        """Basic validation of MLIR syntax. Returns list of errors."""
        errors = []
        lines = code.split("\n")

        brace_count = 0
        paren_count = 0

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue

            # Count braces and parentheses
            brace_count += stripped.count("{") - stripped.count("}")
            paren_count += stripped.count("(") - stripped.count(")")

            # Check for common syntax errors
            if stripped.endswith(","):
                errors.append(f"Line {line_num}: Unexpected trailing comma")

            # Check SSA value format
            ssa_values = [word for word in stripped.split() if word.startswith("%")]
            for ssa in ssa_values:
                if not MLIRUtils.is_valid_ssa_name(ssa.rstrip(",:")):
                    errors.append(f"Line {line_num}: Invalid SSA value name: {ssa}")

        if brace_count != 0:
            errors.append("Mismatched braces in MLIR code")
        if paren_count != 0:
            errors.append("Mismatched parentheses in MLIR code")

        return errors

    @staticmethod
    def get_type_size(mlir_type: str) -> int:
        """Get the size in bits of an MLIR type."""
        if mlir_type.startswith("i"):
            try:
                return int(mlir_type[1:])
            except ValueError:
                return 32  # Default
        elif mlir_type == "f32":
            return 32
        elif mlir_type == "f64":
            return 64
        elif mlir_type.startswith("f"):
            try:
                return int(mlir_type[1:])
            except ValueError:
                return 32
        else:
            return 32  # Default

    @staticmethod
    def is_integer_type(mlir_type: str) -> bool:
        """Check if an MLIR type is an integer type."""
        return mlir_type.startswith("i") and mlir_type[1:].isdigit()

    @staticmethod
    def is_float_type(mlir_type: str) -> bool:
        """Check if an MLIR type is a floating-point type."""
        return mlir_type.startswith("f") and mlir_type[1:].isdigit()

    @staticmethod
    def get_wider_type(type1: str, type2: str) -> str:
        """Get the wider of two MLIR types for promotion."""
        if MLIRUtils.is_float_type(type1) or MLIRUtils.is_float_type(type2):
            # If either is float, promote to float
            if MLIRUtils.is_float_type(type1) and MLIRUtils.is_float_type(type2):
                size1 = MLIRUtils.get_type_size(type1)
                size2 = MLIRUtils.get_type_size(type2)
                return type1 if size1 >= size2 else type2
            elif MLIRUtils.is_float_type(type1):
                return type1
            else:
                return type2
        else:
            # Both are integers, return the wider one
            size1 = MLIRUtils.get_type_size(type1)
            size2 = MLIRUtils.get_type_size(type2)
            return type1 if size1 >= size2 else type2
