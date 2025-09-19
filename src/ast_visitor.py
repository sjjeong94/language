"""
AST Visitor for Python to MLIR conversion

This module implements a visitor pattern to traverse Python AST nodes
and convert them to MLIR operations.
"""

import ast
from typing import Any, Dict, List, Optional, Union
from .mlir_generator import MLIRGenerator


class PythonToMLIRASTVisitor(ast.NodeVisitor):
    """
    AST visitor that converts Python AST nodes to MLIR operations.

    This class uses the visitor pattern to traverse the Python AST and
    generates corresponding MLIR code through the MLIRGenerator.
    """

    def __init__(self):
        self.mlir_generator = MLIRGenerator()
        self.symbol_table: Dict[str, str] = {}  # Maps variable names to MLIR SSA values
        self.current_function: Optional[str] = None
        self.indent_level = 0
        self.imports: Dict[str, str] = {}  # Maps import aliases to module names
        self.oven_lang_alias: Optional[str] = None  # Track oven.language import alias

    def get_mlir_code(self) -> str:
        """Get the generated MLIR code."""
        return self.mlir_generator.get_code()

    def _get_or_create_symbol(self, name: str) -> str:
        """Get or create an SSA value for a symbol."""
        if name not in self.symbol_table:
            self.symbol_table[name] = self.mlir_generator.get_next_ssa_value()
        return self.symbol_table[name]

    def visit_Module(self, node: ast.Module) -> Any:
        """Visit a module node (top-level of Python file)."""
        self.mlir_generator.add_module_header()
        self.generic_visit(node)
        self.mlir_generator.add_module_footer()

    def visit_Import(self, node: ast.Import) -> Any:
        """Visit an import statement."""
        for alias in node.names:
            module_name = alias.name
            alias_name = alias.asname if alias.asname else alias.name
            self.imports[alias_name] = module_name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        """Visit a from...import statement."""
        if node.module == "oven.language":
            for alias in node.names:
                if alias.name == "*":
                    # Handle from oven.language import *
                    self.oven_lang_alias = ""  # No prefix needed
                else:
                    func_name = alias.name
                    alias_name = alias.asname if alias.asname else alias.name
                    # Map the specific function
                    self.imports[alias_name] = f"oven.language.{func_name}"
        elif node.module and "oven.language" in node.module:
            # Handle import oven.language as ol
            for alias in node.names:
                alias_name = alias.asname if alias.asname else alias.name
                self.oven_lang_alias = alias_name

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """Visit a function definition."""
        self.current_function = node.name

        # Check if this is a GPU kernel function (has GPU-specific calls)
        is_gpu_kernel = self._is_gpu_kernel_function(node)
        is_math_function = self._is_math_function(node)
        self._current_is_gpu = is_gpu_kernel  # Track for visit_Return

        if is_gpu_kernel:
            # For GPU kernels, use pointer types for array arguments
            arg_types = ["!llvm.ptr"] * len(node.args.args)
            return_type = ""  # GPU kernels typically don't return values
        elif is_math_function:
            # For math functions, use f32 types
            arg_types = ["f32"] * len(node.args.args)
            return_type = "f32"
        else:
            # Regular functions use i32 types
            arg_types = ["i32"] * len(node.args.args)
            return_type = "i32"

        self._current_return_type = return_type  # Track for visit_Return

        arg_names = [arg.arg for arg in node.args.args]

        # Start function definition
        if is_gpu_kernel:
            self.mlir_generator.start_gpu_function(node.name, arg_types)
        else:
            self.mlir_generator.start_function(node.name, arg_types, return_type)

        # Map arguments to SSA values
        for i, arg_name in enumerate(arg_names):
            if is_gpu_kernel:
                ssa_val = f"%{chr(97+i)}"  # %a, %b, %c, etc. for GPU functions
            else:
                ssa_val = f"%arg{i}"  # %arg0, %arg1, etc. for regular functions
            self.symbol_table[arg_name] = ssa_val

        # Visit function body
        for stmt in node.body:
            self.visit(stmt)

        if is_gpu_kernel:
            self.mlir_generator.add_gpu_return()

        self.mlir_generator.end_function()
        self.current_function = None

    def _is_gpu_kernel_function(self, node: ast.FunctionDef) -> bool:
        """Check if a function contains GPU-specific operations."""
        # GPU functions include both simplified and NVIDIA intrinsic names
        gpu_functions = {
            "get_bdim_x",
            "get_bid_x",
            "get_tid_x",
            "load",
            "store",
            "__nvvm_read_ptx_sreg_ntid_x",
            "__nvvm_read_ptx_sreg_ctaid_x",
            "__nvvm_read_ptx_sreg_tid_x",
            "__load_from_ptr",
            "__store_to_ptr",
        }

        class GPUCallVisitor(ast.NodeVisitor):
            def __init__(self):
                self.has_gpu_calls = False

            def visit_Call(self, node):
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    # Handle ol.load, ol.store etc.
                    if isinstance(node.func.value, ast.Name):
                        module_name = node.func.value.id
                        attr_name = node.func.attr
                        # Check if it's an oven.language call for GPU functions
                        if attr_name in [
                            "load",
                            "store",
                            "get_tid_x",
                            "get_bid_x",
                            "get_bdim_x",
                        ]:
                            func_name = attr_name

                if func_name and func_name in gpu_functions:
                    self.has_gpu_calls = True
                self.generic_visit(node)

        visitor = GPUCallVisitor()
        visitor.visit(node)
        return visitor.has_gpu_calls

    def _is_math_function(self, node: ast.FunctionDef) -> bool:
        """Check if a function contains mathematical operations that need f32 types."""
        math_functions = {"exp", "sigmoid", "sin", "cos", "tan", "sqrt", "log"}

        class MathCallVisitor(ast.NodeVisitor):
            def __init__(self):
                self.has_math_calls = False

            def visit_Call(self, node):
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    # Handle ol.exp, ol.sigmoid etc.
                    if isinstance(node.func.value, ast.Name):
                        module_name = node.func.value.id
                        attr_name = node.func.attr
                        # Check if it's an oven.language call for math functions
                        if attr_name in math_functions:
                            func_name = attr_name

                if func_name and func_name in math_functions:
                    self.has_math_calls = True
                self.generic_visit(node)

        visitor = MathCallVisitor()
        visitor.visit(node)
        return visitor.has_math_calls

    def visit_Return(self, node: ast.Return) -> Any:
        """Visit a return statement."""
        # Check if we're in a GPU kernel function
        is_gpu_kernel = hasattr(self, "_current_is_gpu") and self._current_is_gpu

        if is_gpu_kernel:
            # GPU kernels use plain "return"
            if not node.value:  # Empty return
                # Don't emit anything, GPU return will be handled at function end
                pass
            else:
                value_ssa = self.visit(node.value)
                # GPU kernels typically don't return values, but visit the expression anyway
        else:
            # Regular functions
            if node.value:
                value_ssa = self.visit(node.value)
                return_type = getattr(self, "_current_return_type", "i32")
                self.mlir_generator.add_return(value_ssa, return_type)
            else:
                self.mlir_generator.add_return(None)

    def visit_Assign(self, node: ast.Assign) -> Any:
        """Visit an assignment statement."""
        # Visit the value first
        value_ssa = self.visit(node.value)

        # Handle targets (assume single target for simplicity)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.symbol_table[target.id] = value_ssa

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        """Visit an annotated assignment statement."""
        if node.value:
            value_ssa = self.visit(node.value)
            if isinstance(node.target, ast.Name):
                self.symbol_table[node.target.id] = value_ssa

    def visit_Name(self, node: ast.Name) -> str:
        """Visit a name (variable reference)."""
        if isinstance(node.ctx, ast.Load):
            return self._get_or_create_symbol(node.id)
        return node.id

    def visit_Constant(self, node: ast.Constant) -> str:
        """Visit a constant value."""
        if isinstance(node.value, int):
            return self.mlir_generator.add_constant_int(node.value)
        elif isinstance(node.value, float):
            return self.mlir_generator.add_constant_float(node.value)
        elif isinstance(node.value, str):
            return self.mlir_generator.add_constant_string(node.value)
        elif isinstance(node.value, bool):
            return self.mlir_generator.add_constant_bool(node.value)
        else:
            # Fallback for other constant types
            return self.mlir_generator.add_constant_int(0)

    def visit_BinOp(self, node: ast.BinOp) -> str:
        """Visit a binary operation."""
        left_ssa = self.visit(node.left)
        right_ssa = self.visit(node.right)

        op_map = {
            ast.Add: "arith.addi",
            ast.Sub: "arith.subi",
            ast.Mult: "arith.muli",
            ast.Div: "arith.divsi",
            ast.Mod: "arith.remsi",
            ast.FloorDiv: "arith.divsi",
        }

        op_name = op_map.get(type(node.op), "arith.addi")
        return self.mlir_generator.add_binary_op(op_name, left_ssa, right_ssa)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        """Visit a unary operation."""
        operand_ssa = self.visit(node.operand)

        if isinstance(node.op, ast.UAdd):
            return operand_ssa  # Unary + is a no-op
        elif isinstance(node.op, ast.USub):
            zero_ssa = self.mlir_generator.add_constant_int(0)
            return self.mlir_generator.add_binary_op(
                "arith.subi", zero_ssa, operand_ssa
            )
        elif isinstance(node.op, ast.Not):
            return self.mlir_generator.add_unary_op("arith.xori", operand_ssa)
        else:
            return operand_ssa

    def visit_Compare(self, node: ast.Compare) -> str:
        """Visit a comparison operation."""
        left_ssa = self.visit(node.left)

        # Handle single comparison for simplicity
        if len(node.ops) == 1 and len(node.comparators) == 1:
            right_ssa = self.visit(node.comparators[0])
            op = node.ops[0]

            op_map = {
                ast.Eq: "eq",
                ast.NotEq: "ne",
                ast.Lt: "slt",
                ast.LtE: "sle",
                ast.Gt: "sgt",
                ast.GtE: "sge",
            }

            predicate = op_map.get(type(op), "eq")
            return self.mlir_generator.add_compare_op(predicate, left_ssa, right_ssa)

        return self.mlir_generator.add_constant_bool(True)

    def visit_If(self, node: ast.If) -> Any:
        """Visit an if statement."""
        condition_ssa = self.visit(node.test)

        # Generate labels for the blocks
        then_label = self.mlir_generator.get_next_label("then")
        else_label = self.mlir_generator.get_next_label("else") if node.orelse else None
        end_label = self.mlir_generator.get_next_label("if_end")

        # Conditional branch
        if else_label:
            self.mlir_generator.add_conditional_branch(
                condition_ssa, then_label, else_label
            )
        else:
            self.mlir_generator.add_conditional_branch(
                condition_ssa, then_label, end_label
            )

        # Then block
        self.mlir_generator.add_label(then_label)
        for stmt in node.body:
            self.visit(stmt)
        self.mlir_generator.add_branch(end_label)

        # Else block (if exists)
        if node.orelse and else_label:
            self.mlir_generator.add_label(else_label)
            for stmt in node.orelse:
                self.visit(stmt)
            self.mlir_generator.add_branch(end_label)

        # End block
        self.mlir_generator.add_label(end_label)

    def visit_While(self, node: ast.While) -> Any:
        """Visit a while loop."""
        loop_header = self.mlir_generator.get_next_label("loop_header")
        loop_body = self.mlir_generator.get_next_label("loop_body")
        loop_end = self.mlir_generator.get_next_label("loop_end")

        # Jump to loop header
        self.mlir_generator.add_branch(loop_header)

        # Loop header (condition check)
        self.mlir_generator.add_label(loop_header)
        condition_ssa = self.visit(node.test)
        self.mlir_generator.add_conditional_branch(condition_ssa, loop_body, loop_end)

        # Loop body
        self.mlir_generator.add_label(loop_body)
        for stmt in node.body:
            self.visit(stmt)
        self.mlir_generator.add_branch(loop_header)

        # Loop end
        self.mlir_generator.add_label(loop_end)

    def visit_For(self, node: ast.For) -> Any:
        """Visit a for loop (simplified implementation)."""
        # This is a simplified implementation
        # In a real compiler, you'd need to handle iterables properly
        loop_body = self.mlir_generator.get_next_label("for_body")
        loop_end = self.mlir_generator.get_next_label("for_end")

        # For simplicity, treat as while loop with manual iteration
        self.mlir_generator.add_label(loop_body)
        for stmt in node.body:
            self.visit(stmt)
        self.mlir_generator.add_branch(loop_end)

        self.mlir_generator.add_label(loop_end)

    def visit_Call(self, node: ast.Call) -> str:
        """Visit a function call."""
        # Get function name - handle both direct calls and attribute access
        func_name = None
        module_prefix = None

        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # Handle module.function calls (e.g., ol.load)
            if isinstance(node.func.value, ast.Name):
                module_prefix = node.func.value.id
                func_name = node.func.attr

                # Check if this is an oven.language call
                if (
                    module_prefix in self.imports
                    and self.imports[module_prefix] == "oven.language"
                ):
                    func_name = f"oven_lang_{func_name}"
                elif module_prefix == self.oven_lang_alias:
                    func_name = f"oven_lang_{func_name}"

        if not func_name:
            func_name = "unknown"

        # Handle GPU-specific function calls for kernel.py support
        # Support both direct calls and oven.language module calls
        if func_name in ["get_bdim_x", "oven_lang_get_bdim_x"]:
            return self.mlir_generator.add_gpu_block_dim_x()
        elif func_name in ["get_bid_x", "oven_lang_get_bid_x"]:
            return self.mlir_generator.add_gpu_block_id_x()
        elif func_name in ["get_tid_x", "oven_lang_get_tid_x"]:
            return self.mlir_generator.add_gpu_thread_id_x()
        elif func_name == "__nvvm_read_ptx_sreg_ntid_x":
            return self.mlir_generator.add_gpu_block_dim_x()
        elif func_name == "__nvvm_read_ptx_sreg_ctaid_x":
            return self.mlir_generator.add_gpu_block_id_x()
        elif func_name == "__nvvm_read_ptx_sreg_tid_x":
            return self.mlir_generator.add_gpu_thread_id_x()
        elif func_name in ["load", "oven_lang_load"]:
            # load(ptr, offset)
            if len(node.args) >= 2:
                ptr_ssa = self.visit(node.args[0])
                offset_ssa = self.visit(node.args[1])
                return self.mlir_generator.add_gpu_load(ptr_ssa, offset_ssa)
        elif func_name == "__load_from_ptr":
            # __load_from_ptr(ptr, offset)
            if len(node.args) >= 2:
                ptr_ssa = self.visit(node.args[0])
                offset_ssa = self.visit(node.args[1])
                return self.mlir_generator.add_gpu_load(ptr_ssa, offset_ssa)
        elif func_name in ["store", "oven_lang_store"]:
            # store(value, ptr, offset)
            if len(node.args) >= 3:
                value_ssa = self.visit(node.args[0])
                ptr_ssa = self.visit(node.args[1])
                offset_ssa = self.visit(node.args[2])
                self.mlir_generator.add_gpu_store(value_ssa, ptr_ssa, offset_ssa)
                return value_ssa  # Return the stored value as SSA
        elif func_name == "__store_to_ptr":
            # __store_to_ptr(ptr, offset, value)
            if len(node.args) >= 3:
                ptr_ssa = self.visit(node.args[0])
                offset_ssa = self.visit(node.args[1])
                value_ssa = self.visit(node.args[2])
                self.mlir_generator.add_gpu_store(value_ssa, ptr_ssa, offset_ssa)
                return value_ssa  # Return the stored value as SSA
        # Handle mathematical function calls
        elif func_name in ["exp", "oven_lang_exp"]:
            # exp(value)
            if len(node.args) >= 1:
                operand_ssa = self.visit(node.args[0])
                return self.mlir_generator.add_math_exp(operand_ssa)
        elif func_name in ["sigmoid", "oven_lang_sigmoid"]:
            # sigmoid(value)
            if len(node.args) >= 1:
                operand_ssa = self.visit(node.args[0])
                return self.mlir_generator.add_oven_sigmoid(operand_ssa)

        # Visit arguments for regular function calls
        arg_ssa_values = []
        for arg in node.args:
            arg_ssa_values.append(self.visit(arg))

        return self.mlir_generator.add_function_call(func_name, arg_ssa_values)

    def visit_Expr(self, node: ast.Expr) -> Any:
        """Visit an expression statement."""
        self.visit(node.value)

    def generic_visit(self, node: ast.AST) -> Any:
        """Generic visit method for unsupported nodes."""
        # For unsupported nodes, just visit children
        super().generic_visit(node)
