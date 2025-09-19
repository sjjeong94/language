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

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """Visit a function definition."""
        self.current_function = node.name

        # Extract argument types (assuming all are i32 for simplicity)
        arg_types = ["i32"] * len(node.args.args)
        arg_names = [arg.arg for arg in node.args.args]

        # Start function definition
        self.mlir_generator.start_function(node.name, arg_types, "i32")

        # Map arguments to SSA values
        for i, arg_name in enumerate(arg_names):
            ssa_val = f"%arg{i}"
            self.symbol_table[arg_name] = ssa_val

        # Visit function body
        for stmt in node.body:
            self.visit(stmt)

        self.mlir_generator.end_function()
        self.current_function = None

    def visit_Return(self, node: ast.Return) -> Any:
        """Visit a return statement."""
        if node.value:
            value_ssa = self.visit(node.value)
            self.mlir_generator.add_return(value_ssa)
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
        # Get function name
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        else:
            func_name = "unknown"

        # Visit arguments
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
