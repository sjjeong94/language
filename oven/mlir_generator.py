"""
MLIR Code Generator

This module generates MLIR code from Python AST nodes.
Provides utilities for creating MLIR operations, blocks, and functions.
"""

from typing import List, Optional, Dict, Any
from .utils.mlir_utils import MLIRUtils


class MLIRGenerator:
    """
    Generates MLIR code from Python AST operations.

    This class provides methods to generate various MLIR constructs
    including operations, basic blocks, functions, and modules.
    """

    def __init__(self):
        self.code_lines: List[str] = []
        self.ssa_counter = 0
        self.label_counter = 0
        self.indent_level = 0
        self.utils = MLIRUtils()

    def _emit(self, line: str, extra_indent: int = 0) -> None:
        """Emit a line of MLIR code with proper indentation."""
        indent = "  " * (self.indent_level + extra_indent)
        self.code_lines.append(f"{indent}{line}")

    def get_next_ssa_value(self) -> str:
        """Generate the next SSA value name."""
        value = f"%{self.ssa_counter}"
        self.ssa_counter += 1
        return value

    def get_next_label(self, prefix: str = "label") -> str:
        """Generate the next basic block label."""
        label = f"^{prefix}{self.label_counter}"
        self.label_counter += 1
        return label

    def get_code(self) -> str:
        """Get the generated MLIR code as a string."""
        return "\n".join(self.code_lines)

    def add_module_header(self) -> None:
        """Add MLIR module header."""
        self._emit("// Generated MLIR code from Python source")
        self._emit("")

    def add_module_footer(self) -> None:
        """Add MLIR module footer."""
        self._emit("")

    def start_function(self, name: str, arg_types: List[str], return_type: str) -> None:
        """Start a function definition."""
        args_str = ", ".join(
            f"%arg{i}: {arg_type}" for i, arg_type in enumerate(arg_types)
        )
        self._emit(f"func.func @{name}({args_str}) -> {return_type} {{")
        self.indent_level += 1

    def start_gpu_function(self, name: str, arg_types: List[str]) -> None:
        """Start a GPU function definition (no return type)."""
        args_str = ", ".join(
            f"%{chr(97+i)}: {arg_type}" for i, arg_type in enumerate(arg_types)
        )
        self._emit(f"func.func @{name}({args_str}) {{")
        self.indent_level += 1

    def end_function(self) -> None:
        """End a function definition."""
        self.indent_level -= 1
        self._emit("}")
        self._emit("")

    def add_constant_int(self, value: int, bit_width: int = 32) -> str:
        """Add an integer constant operation."""
        ssa_val = self.get_next_ssa_value()
        self._emit(f"{ssa_val} = arith.constant {value} : i{bit_width}")
        return ssa_val

    def add_constant_float(self, value: float, precision: str = "f32") -> str:
        """Add a floating-point constant operation."""
        ssa_val = self.get_next_ssa_value()
        self._emit(f"{ssa_val} = arith.constant {value} : {precision}")
        return ssa_val

    def add_constant_bool(self, value: bool) -> str:
        """Add a boolean constant operation."""
        ssa_val = self.get_next_ssa_value()
        bool_val = "true" if value else "false"
        self._emit(f"{ssa_val} = arith.constant {bool_val}")
        return ssa_val

    def add_constant_string(self, value: str) -> str:
        """Add a string constant operation (simplified)."""
        ssa_val = self.get_next_ssa_value()
        escaped_value = value.replace('"', '\\"')
        self._emit(f'{ssa_val} = llvm.mlir.constant("{escaped_value}") : !llvm.ptr<i8>')
        return ssa_val

    def add_binary_op(
        self, op_name: str, lhs: str, rhs: str, result_type: str = "i32"
    ) -> str:
        """Add a binary operation."""
        ssa_val = self.get_next_ssa_value()
        self._emit(f"{ssa_val} = {op_name} {lhs}, {rhs} : {result_type}")
        return ssa_val

    def add_unary_op(self, op_name: str, operand: str, result_type: str = "i32") -> str:
        """Add a unary operation."""
        ssa_val = self.get_next_ssa_value()
        if op_name == "arith.xori":
            # For NOT operation, XOR with all 1s
            true_val = self.add_constant_int(-1)  # All 1s for bitwise NOT
            self._emit(f"{ssa_val} = {op_name} {operand}, {true_val} : {result_type}")
        else:
            self._emit(f"{ssa_val} = {op_name} {operand} : {result_type}")
        return ssa_val

    def add_compare_op(
        self, predicate: str, lhs: str, rhs: str, operand_type: str = "i32"
    ) -> str:
        """Add a comparison operation."""
        ssa_val = self.get_next_ssa_value()
        self._emit(f"{ssa_val} = arith.cmpi {predicate}, {lhs}, {rhs} : {operand_type}")
        return ssa_val

    def add_return(self, value: Optional[str] = None, return_type: str = "i32") -> None:
        """Add a return operation."""
        if value:
            self._emit(f"func.return {value} : {return_type}")
        else:
            self._emit("func.return")

    def add_label(self, label: str) -> None:
        """Add a basic block label."""
        # Remove the ^ prefix for the label definition
        clean_label = label[1:] if label.startswith("^") else label
        self.indent_level -= 1
        self._emit(f"{clean_label}:")
        self.indent_level += 1

    def add_branch(self, target_label: str) -> None:
        """Add an unconditional branch."""
        self._emit(f"cf.br {target_label}")

    def add_conditional_branch(
        self, condition: str, true_label: str, false_label: str
    ) -> None:
        """Add a conditional branch."""
        self._emit(f"cf.cond_br {condition}, {true_label}, {false_label}")

    def add_function_call(
        self, func_name: str, args: List[str], return_type: str = "i32"
    ) -> str:
        """Add a function call operation."""
        ssa_val = self.get_next_ssa_value()
        args_str = ", ".join(args)

        if args:
            arg_types = ", ".join(["i32"] * len(args))  # Assume all i32 for simplicity
            self._emit(
                f"{ssa_val} = func.call @{func_name}({args_str}) : ({arg_types}) -> {return_type}"
            )
        else:
            self._emit(f"{ssa_val} = func.call @{func_name}() : () -> {return_type}")

        return ssa_val

    def add_load(self, address: str, element_type: str = "i32") -> str:
        """Add a load operation."""
        ssa_val = self.get_next_ssa_value()
        self._emit(f"{ssa_val} = memref.load {address}[] : memref<{element_type}>")
        return ssa_val

    def add_store(self, value: str, address: str, element_type: str = "i32") -> None:
        """Add a store operation."""
        self._emit(f"memref.store {value}, {address}[] : memref<{element_type}>")

    def add_alloca(self, element_type: str = "i32") -> str:
        """Add an alloca operation for local variables."""
        ssa_val = self.get_next_ssa_value()
        self._emit(f"{ssa_val} = memref.alloca() : memref<{element_type}>")
        return ssa_val

    def add_comment(self, comment: str) -> None:
        """Add a comment to the MLIR code."""
        self._emit(f"// {comment}")

    def add_cast(self, value: str, from_type: str, to_type: str) -> str:
        """Add a type casting operation."""
        ssa_val = self.get_next_ssa_value()

        # Choose appropriate cast operation based on types
        if from_type.startswith("i") and to_type.startswith("i"):
            # Integer to integer cast
            from_bits = int(from_type[1:])
            to_bits = int(to_type[1:])

            if from_bits < to_bits:
                # Sign extend
                self._emit(
                    f"{ssa_val} = arith.extsi {value} : {from_type} to {to_type}"
                )
            elif from_bits > to_bits:
                # Truncate
                self._emit(
                    f"{ssa_val} = arith.trunci {value} : {from_type} to {to_type}"
                )
            else:
                # Same size, no cast needed
                return value
        elif from_type.startswith("f") and to_type.startswith("f"):
            # Float to float cast
            self._emit(f"{ssa_val} = arith.fpext {value} : {from_type} to {to_type}")
        elif from_type.startswith("i") and to_type.startswith("f"):
            # Integer to float
            self._emit(f"{ssa_val} = arith.sitofp {value} : {from_type} to {to_type}")
        elif from_type.startswith("f") and to_type.startswith("i"):
            # Float to integer
            self._emit(f"{ssa_val} = arith.fptosi {value} : {from_type} to {to_type}")
        else:
            # Generic cast
            self._emit(
                f"{ssa_val} = builtin.unrealized_conversion_cast {value} : {from_type} to {to_type}"
            )

        return ssa_val

    def add_phi_node(
        self, values_and_blocks: List[tuple], result_type: str = "i32"
    ) -> str:
        """Add a phi node for SSA form."""
        ssa_val = self.get_next_ssa_value()

        # In MLIR, phi nodes are typically handled by block arguments
        # This is a simplified representation
        pairs = ", ".join(f"({value} : {block})" for value, block in values_and_blocks)
        self._emit(f"{ssa_val} = phi {pairs} : {result_type}")
        return ssa_val

    # GPU-related operations for kernel.py support
    def add_gpu_block_dim_x(self) -> str:
        """Add NVVM read block dimension X operation."""
        ssa_val = self.get_next_ssa_value()
        self._emit(f"{ssa_val} = nvvm.read.ptx.sreg.ntid.x : i32")
        return ssa_val

    def add_gpu_block_id_x(self) -> str:
        """Add NVVM read block ID X operation."""
        ssa_val = self.get_next_ssa_value()
        self._emit(f"{ssa_val} = nvvm.read.ptx.sreg.ctaid.x : i32")
        return ssa_val

    def add_gpu_thread_id_x(self) -> str:
        """Add NVVM read thread ID X operation."""
        ssa_val = self.get_next_ssa_value()
        self._emit(f"{ssa_val} = nvvm.read.ptx.sreg.tid.x : i32")
        return ssa_val

    def add_gpu_thread_id_y(self) -> str:
        """Add NVVM read thread ID Y operation."""
        ssa_val = self.get_next_ssa_value()
        self._emit(f"{ssa_val} = nvvm.read.ptx.sreg.tid.y : i32")
        return ssa_val

    def add_gpu_block_id_y(self) -> str:
        """Add NVVM read block ID Y operation."""
        ssa_val = self.get_next_ssa_value()
        self._emit(f"{ssa_val} = nvvm.read.ptx.sreg.ctaid.y : i32")
        return ssa_val

    def add_gpu_load(self, ptr: str, offset: str, result_type: str = "f32") -> str:
        """Add GPU load operation."""
        ssa_val = self.get_next_ssa_value()
        self._emit(
            f"{ssa_val} = oven.load {ptr}, {offset} : (!llvm.ptr, i32) -> {result_type}"
        )
        return ssa_val

    def add_gpu_store(
        self, value: str, ptr: str, offset: str, value_type: str = "f32"
    ) -> None:
        """Add GPU store operation."""
        self._emit(
            f"oven.store {value}, {ptr}, {offset} : ({value_type}, !llvm.ptr, i32)"
        )

    def add_gpu_return(self) -> None:
        """Add GPU return operation."""
        self._emit("return")

    # Mathematical operations
    def add_math_exp(self, operand: str, operand_type: str = "f32") -> str:
        """Add math.exp operation."""
        ssa_val = self.get_next_ssa_value()
        self._emit(f"{ssa_val} = math.exp {operand} : {operand_type}")
        return ssa_val

    def add_oven_sigmoid(self, operand: str, operand_type: str = "f32") -> str:
        """Add oven.sigmoid operation."""
        ssa_val = self.get_next_ssa_value()
        self._emit(
            f"{ssa_val} = oven.sigmoid {operand} : {operand_type} -> {operand_type}"
        )
        return ssa_val

    # Arithmetic operations
    def add_arith_muli(self, lhs: str, rhs: str, result_type: str = "i32") -> str:
        """Add integer multiplication operation."""
        ssa_val = self.get_next_ssa_value()
        self._emit(f"{ssa_val} = arith.muli {lhs}, {rhs} : {result_type}")
        return ssa_val

    def add_arith_addi(self, lhs: str, rhs: str, result_type: str = "i32") -> str:
        """Add integer addition operation."""
        ssa_val = self.get_next_ssa_value()
        self._emit(f"{ssa_val} = arith.addi {lhs}, {rhs} : {result_type}")
        return ssa_val

    def add_arith_mulf(self, lhs: str, rhs: str, result_type: str = "f32") -> str:
        """Add floating-point multiplication operation."""
        ssa_val = self.get_next_ssa_value()
        self._emit(f"{ssa_val} = arith.mulf {lhs}, {rhs} : {result_type}")
        return ssa_val

    def add_arith_addf(self, lhs: str, rhs: str, result_type: str = "f32") -> str:
        """Add floating-point addition operation."""
        ssa_val = self.get_next_ssa_value()
        self._emit(f"{ssa_val} = arith.addf {lhs}, {rhs} : {result_type}")
        return ssa_val

    def add_arith_subi(self, lhs: str, rhs: str, result_type: str = "i32") -> str:
        """Add integer subtraction operation."""
        ssa_val = self.get_next_ssa_value()
        self._emit(f"{ssa_val} = arith.subi {lhs}, {rhs} : {result_type}")
        return ssa_val

    def add_arith_subf(self, lhs: str, rhs: str, result_type: str = "f32") -> str:
        """Add floating-point subtraction operation."""
        ssa_val = self.get_next_ssa_value()
        self._emit(f"{ssa_val} = arith.subf {lhs}, {rhs} : {result_type}")
        return ssa_val

    def add_arith_divf(self, lhs: str, rhs: str, result_type: str = "f32") -> str:
        """Add floating-point division operation."""
        ssa_val = self.get_next_ssa_value()
        self._emit(f"{ssa_val} = arith.divf {lhs}, {rhs} : {result_type}")
        return ssa_val

    # Type conversion
    def add_arith_index_cast(self, value: str, from_type: str, to_type: str) -> str:
        """Add index cast operation."""
        ssa_val = self.get_next_ssa_value()
        self._emit(f"{ssa_val} = arith.index_cast {value} : {from_type} to {to_type}")
        return ssa_val

    # Constants for different types
    def add_constant_index(self, value: int) -> str:
        """Add an index constant operation."""
        ssa_val = self.get_next_ssa_value()
        self._emit(f"{ssa_val} = arith.constant {value} : index")
        return ssa_val

    # Loop operations
    def add_scf_for(
        self, start: str, end: str, step: str, iter_args: list = None
    ) -> str:
        """Add SCF for loop operation."""
        if iter_args is None:
            iter_args = []

        ssa_val = self.get_next_ssa_value()
        iter_args_str = ", ".join(iter_args) if iter_args else ""
        iter_types = " -> (f32)" if iter_args else ""

        if iter_args:
            self._emit(
                f"{ssa_val} = scf.for {start} = {start} to {end} step {step} iter_args({iter_args_str}){iter_types} {{"
            )
        else:
            self._emit(f"scf.for {start} = {start} to {end} step {step} {{")

        self.indent_level += 1
        return ssa_val

    def add_scf_yield(self, values: list = None) -> None:
        """Add SCF yield operation."""
        if values is None:
            values = []

        if values:
            values_str = ", ".join(values)
            types_str = " : " + ", ".join(["f32"] * len(values))
            self._emit(f"scf.yield {values_str}{types_str}")
        else:
            self._emit("scf.yield")

    def end_scf_for(self) -> None:
        """End SCF for loop."""
        self.indent_level -= 1
        self._emit("}")
