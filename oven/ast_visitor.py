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
        self.symbol_types: Dict[str, str] = {}  # Maps variable names to MLIR types
        self.ssa_types: Dict[str, str] = {}  # Maps SSA values to their types
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

    def _get_symbol_type(self, name: str) -> str:
        """Get the MLIR type of a symbol."""
        if name in self.symbol_types:
            return self.symbol_types[name]
        # Default fallback
        return "i32"

    def _infer_type_from_ssa(self, ssa_val: str) -> str:
        """Infer type from SSA value by looking up in symbol tables."""
        # First check direct SSA type mapping
        if ssa_val in self.ssa_types:
            return self.ssa_types[ssa_val]

        # Find the variable name that maps to this SSA value
        for var_name, var_ssa in self.symbol_table.items():
            if var_ssa == ssa_val:
                return self._get_symbol_type(var_name)
        # Default fallback
        return "i32"

    def _track_ssa_type(self, ssa_val: str, ssa_type: str) -> None:
        """Track the type of an SSA value."""
        if ssa_val:
            self.ssa_types[ssa_val] = ssa_type

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
        self._current_is_math = is_math_function  # Track for arithmetic operations

        # Get argument types from type annotations or infer from context
        arg_types = self._get_argument_types(node)

        # Get return type from type annotation or infer from context
        return_type = self._get_return_type(node, is_gpu_kernel, is_math_function)

        self._current_return_type = return_type  # Track for visit_Return

        arg_names = [arg.arg for arg in node.args.args]

        # Start function definition
        if is_gpu_kernel:
            self.mlir_generator.start_gpu_function(node.name, arg_types, arg_names)
        else:
            self.mlir_generator.start_function(node.name, arg_types, return_type)

        # Map arguments to SSA values and track their types
        for i, (arg_name, arg_type) in enumerate(zip(arg_names, arg_types)):
            if is_gpu_kernel:
                ssa_val = f"%{arg_name}"  # Use actual parameter names for GPU functions
            else:
                ssa_val = f"%arg{i}"  # %arg0, %arg1, etc. for regular functions
            self.symbol_table[arg_name] = ssa_val
            self.symbol_types[arg_name] = arg_type

        # Visit function body (skip docstring)
        for i, stmt in enumerate(node.body):
            # Skip docstring (first statement if it's a string literal)
            if (
                i == 0
                and isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, (ast.Str, ast.Constant))
                and isinstance(
                    getattr(stmt.value, "value", getattr(stmt.value, "s", None)), str
                )
            ):
                continue
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
            "get_bid_y",
            "get_tid_x",
            "get_tid_y",
            "load",
            "store",
            "vload",
            "vstore",
            "smem",
            "barrier",
            "load_input_x",
            "load_input_y",
            "store_output_x",
            "store_output_y",
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
                            "vload",
                            "vstore",
                            "smem",
                            "barrier",
                            "nvvm_read_ptx_sreg_ntid_x",
                            "nvvm_read_ptx_sreg_ctaid_x",
                            "nvvm_read_ptx_sreg_tid_x",
                            "load_input_x",
                            "load_input_y",
                            "store_output_x",
                            "store_output_y",
                            "get_tid_x",
                            "get_tid_y",
                            "get_bid_x",
                            "get_bid_y",
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
        math_functions = {
            "exp",
            "exp2",
            "sigmoid",
            "sin",
            "cos",
            "tan",
            "sqrt",
            "log",
            "log2",
        }

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

    def _parse_type_annotation(self, annotation) -> str:
        """Parse a Python type annotation to MLIR type string."""
        if annotation is None:
            return None

        # Handle simple name annotations (int, f32, etc.)
        if isinstance(annotation, ast.Name):
            type_name = annotation.id
            type_mapping = {"int": "i32", "i32": "i32", "f32": "f32", "index": "index"}
            return type_mapping.get(type_name, "i32")  # Default to i32

        # Handle attribute annotations (ol.ptr, ol.f32, etc.)
        elif isinstance(annotation, ast.Attribute):
            if isinstance(annotation.value, ast.Name):
                module_name = annotation.value.id
                attr_name = annotation.attr

                # Handle oven.language types (ol.ptr, ol.f32, etc.)
                if module_name == "ol":
                    type_mapping = {
                        "ptr": "!llvm.ptr",
                        "f32": "f32",
                        "i32": "i32",
                        "index": "index",
                    }
                    return type_mapping.get(attr_name, "i32")

        # Default fallback
        return "i32"

    def _get_argument_types(self, node: ast.FunctionDef) -> List[str]:
        """Get argument types from type annotations or infer from context."""
        arg_types = []

        # Check if this is a GPU kernel function or math function
        is_gpu_kernel = self._is_gpu_kernel_function(node)
        is_math_function = self._is_math_function(node)

        for arg in node.args.args:
            # Try to get type from annotation first
            if arg.annotation:
                mlir_type = self._parse_type_annotation(arg.annotation)
                arg_types.append(mlir_type)
            else:
                # Fall back to context-based inference
                if is_gpu_kernel:
                    arg_types.append("!llvm.ptr")
                elif is_math_function:
                    arg_types.append("f32")
                else:
                    arg_types.append("i32")

        return arg_types

    def _get_return_type(
        self, node: ast.FunctionDef, is_gpu_kernel: bool, is_math_function: bool
    ) -> str:
        """Get return type from type annotation or infer from context."""
        # Try to get type from annotation first
        if node.returns:
            return self._parse_type_annotation(node.returns)

        # Fall back to context-based inference
        if is_gpu_kernel:
            return ""  # GPU kernels typically don't return values
        elif is_math_function:
            return "f32"
        else:
            return "i32"

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
            ssa_val = self.mlir_generator.add_constant_int(node.value)
            self._track_ssa_type(ssa_val, "i32")
            return ssa_val
        elif isinstance(node.value, float):
            ssa_val = self.mlir_generator.add_constant_float(node.value)
            self._track_ssa_type(ssa_val, "f32")
            return ssa_val
        elif isinstance(node.value, str):
            ssa_val = self.mlir_generator.add_constant_string(node.value)
            self._track_ssa_type(ssa_val, "!llvm.ptr")
            return ssa_val
        elif isinstance(node.value, bool):
            ssa_val = self.mlir_generator.add_constant_bool(node.value)
            self._track_ssa_type(ssa_val, "i1")
            return ssa_val
        else:
            # Fallback for other constant types
            ssa_val = self.mlir_generator.add_constant_int(0)
            self._track_ssa_type(ssa_val, "i32")
            return ssa_val

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
        """Visit a for loop and convert to SCF for loop."""
        # Handle range() function calls specifically
        if (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        ):

            # Extract range arguments (start, stop, step)
            args = node.iter.args
            if len(args) == 1:
                # range(stop)
                start_const = 0
                end_val = self.visit(args[0])
                step_const = 1
            elif len(args) == 2:
                # range(start, stop)
                start_const = args[0].value if isinstance(args[0], ast.Constant) else 0
                end_val = self.visit(args[1])
                step_const = 1
            elif len(args) == 3:
                # range(start, stop, step)
                start_const = args[0].value if isinstance(args[0], ast.Constant) else 0
                end_val = self.visit(args[1])
                if isinstance(args[2], ast.Constant):
                    step_const = args[2].value
                else:
                    # Handle variable step
                    step_val = self.visit(args[2])
                    step_const = None
            else:
                raise ValueError("Invalid range() arguments")

            # Create index constants
            start_index = self.mlir_generator.add_constant_index(start_const)
            end_index = self.mlir_generator.add_arith_index_cast(
                end_val, "i32", "index"
            )
            if step_const is not None:
                step_index = self.mlir_generator.add_constant_index(step_const)
            else:
                # Convert variable step to index
                step_index = self.mlir_generator.add_arith_index_cast(
                    step_val, "i32", "index"
                )

            # Find accumulator variables - variables assigned in loop that exist before loop
            accum_vars = []

            def find_assignments(node_list, depth=0):
                """Recursively find all assignment targets in a node list."""
                assignments = []
                for stmt in node_list:
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Name):
                                assignments.append((target.id, depth))
                    elif isinstance(stmt, ast.For):
                        # Recursively check nested for loops with increased depth
                        assignments.extend(find_assignments(stmt.body, depth + 1))
                    elif hasattr(stmt, "body"):
                        # Check other compound statements
                        assignments.extend(find_assignments(stmt.body, depth))
                return assignments

            all_assignments = find_assignments(node.body)

            # Include variables that are assigned at any depth and exist in symbol table
            for var_name, depth in all_assignments:
                if var_name in self.symbol_table and var_name not in accum_vars:
                    accum_vars.append(var_name)

            # Debug: print what we found (simplified)
            # print(f"For loop target: {node.target.id}, accumulator variables: {accum_vars}")

            # Get initial values for accumulator variables
            iter_args = []
            for var_name in accum_vars:
                iter_args.append(self.symbol_table[var_name])

            # print(f"iter_args for {node.target.id}: {iter_args}")

            # Generate SCF for loop
            loop_var = node.target.id
            if iter_args:
                # Reserve SSA value for iter_arg (will be used inside the loop)
                iter_arg_ssa = self.mlir_generator.ssa_counter
                self.mlir_generator.ssa_counter += 1  # Reserve this number

                iter_arg_pairs = [
                    (str(iter_arg_ssa), iter_args[0])  # Use reserved SSA counter
                ]  # Use SSA naming convention
                loop_ssa = self.mlir_generator.add_scf_for(
                    f"{loop_var}_index",
                    start_index,
                    end_index,
                    step_index,
                    iter_arg_pairs,
                )
            else:
                self.mlir_generator.add_scf_for(
                    f"{loop_var}_index", start_index, end_index, step_index
                )
                loop_ssa = None

            # Save symbol table
            old_symbols = self.symbol_table.copy()

            # Set up loop variable (convert from index to i32)
            loop_var = node.target.id
            loop_var_ssa = self.mlir_generator.add_arith_index_cast(
                f"%{loop_var}_index", "index", "i32"
            )
            self.symbol_table[loop_var] = loop_var_ssa

            # Set up accumulator variables as block arguments
            for i, var_name in enumerate(accum_vars):
                # In SCF for, iter_args become block arguments with SSA names
                if iter_args:
                    # Use the reserved SSA counter that was used for iter_arg
                    self.symbol_table[var_name] = f"%{iter_arg_ssa}"
                else:
                    self.symbol_table[var_name] = f"%{var_name}"

            # Process loop body
            nested_loop_results = []  # Track results from nested loops
            accumulated_values = {}  # Track values for accumulator variables
            nested_vars = set()  # Track variables defined inside nested loops

            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    # Check if this assignment involves variables from nested loops
                    if isinstance(stmt.value, ast.BinOp):
                        # Check if the right operand is a nested loop variable
                        right_var = None
                        if isinstance(stmt.value.right, ast.Name):
                            right_var = stmt.value.right.id

                        # If the assignment uses a nested loop variable, skip it
                        # because we already handled it when processing the nested loop
                        if right_var and right_var in nested_vars:
                            continue

                    result_ssa = self.visit(stmt.value)
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            var_name = target.id
                            self.symbol_table[var_name] = result_ssa
                            if var_name in accum_vars:
                                accumulated_values[var_name] = result_ssa
                elif isinstance(stmt, ast.For):
                    # Track variables assigned in this nested loop
                    def find_nested_vars(nested_node):
                        vars_in_nested = set()
                        for nested_stmt in nested_node.body:
                            if isinstance(nested_stmt, ast.Assign):
                                for target in nested_stmt.targets:
                                    if isinstance(target, ast.Name):
                                        vars_in_nested.add(target.id)
                        return vars_in_nested

                    nested_loop_vars = find_nested_vars(stmt)
                    nested_vars.update(nested_loop_vars)

                    # This is a nested for loop
                    result_ssa = self.visit(stmt)
                    if result_ssa:
                        nested_loop_results.append(result_ssa)
                        # The nested loop result is already the updated accumulator value
                        # No need to add it again - just update the symbol table
                        for var_name in accum_vars:
                            self.symbol_table[var_name] = result_ssa
                            accumulated_values[var_name] = result_ssa
                else:
                    self.visit(stmt)

            # Generate yield if we have iter_args
            if iter_args:
                # For iter_args, we MUST yield values
                # Use the current values in symbol table for accumulator variables
                yield_values = []
                for var_name in accum_vars:
                    current_val = self.symbol_table.get(var_name)
                    if current_val:
                        yield_values.append(current_val)

                # Always emit yield for iter_args loops
                self.mlir_generator.add_scf_yield(yield_values)

            # End for loop
            self.mlir_generator.end_scf_for()

            # Update symbol table with loop results
            result_ssa = None
            if loop_ssa:
                for var_name in accum_vars:
                    self.symbol_table[var_name] = loop_ssa
                result_ssa = loop_ssa

            # Clean up - remove loop variable, keep updated accumulators
            if loop_var in self.symbol_table:
                del self.symbol_table[loop_var]

            return result_ssa

        else:
            # Fallback for non-range iterables
            loop_body = self.mlir_generator.get_next_label("for_body")
            loop_end = self.mlir_generator.get_next_label("for_end")

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
            ssa_val = self.mlir_generator.add_gpu_block_dim_x()
            self._track_ssa_type(ssa_val, "i32")
            return ssa_val
        elif func_name in ["get_bid_x", "oven_lang_get_bid_x"]:
            ssa_val = self.mlir_generator.add_gpu_block_id_x()
            self._track_ssa_type(ssa_val, "i32")
            return ssa_val
        elif func_name in ["get_bid_y", "oven_lang_get_bid_y"]:
            ssa_val = self.mlir_generator.add_gpu_block_id_y()
            self._track_ssa_type(ssa_val, "i32")
            return ssa_val
        elif func_name in ["get_tid_x", "oven_lang_get_tid_x"]:
            ssa_val = self.mlir_generator.add_gpu_thread_id_x()
            self._track_ssa_type(ssa_val, "i32")
            return ssa_val
        elif func_name in ["get_tid_y", "oven_lang_get_tid_y"]:
            ssa_val = self.mlir_generator.add_gpu_thread_id_y()
            self._track_ssa_type(ssa_val, "i32")
            return ssa_val
        elif func_name == "__nvvm_read_ptx_sreg_ntid_x":
            ssa_val = self.mlir_generator.add_gpu_block_dim_x()
            self._track_ssa_type(ssa_val, "i32")
            return ssa_val
        elif func_name in [
            "nvvm_read_ptx_sreg_ntid_x",
            "oven_lang_nvvm_read_ptx_sreg_ntid_x",
        ]:
            ssa_val = self.mlir_generator.add_gpu_block_dim_x()
            self._track_ssa_type(ssa_val, "i32")
            return ssa_val
        elif func_name == "__nvvm_read_ptx_sreg_ctaid_x":
            ssa_val = self.mlir_generator.add_gpu_block_id_x()
            self._track_ssa_type(ssa_val, "i32")
            return ssa_val
        elif func_name in [
            "nvvm_read_ptx_sreg_ctaid_x",
            "oven_lang_nvvm_read_ptx_sreg_ctaid_x",
        ]:
            ssa_val = self.mlir_generator.add_gpu_block_id_x()
            self._track_ssa_type(ssa_val, "i32")
            return ssa_val
        elif func_name == "__nvvm_read_ptx_sreg_tid_x":
            ssa_val = self.mlir_generator.add_gpu_thread_id_x()
            self._track_ssa_type(ssa_val, "i32")
            return ssa_val
        elif func_name in [
            "nvvm_read_ptx_sreg_tid_x",
            "oven_lang_nvvm_read_ptx_sreg_tid_x",
        ]:
            ssa_val = self.mlir_generator.add_gpu_thread_id_x()
            self._track_ssa_type(ssa_val, "i32")
            return ssa_val
        elif func_name in ["load", "oven_lang_load"]:
            # load(ptr, offset)
            if len(node.args) >= 2:
                ptr_ssa = self.visit(node.args[0])
                offset_ssa = self.visit(node.args[1])

                # Check if the pointer is a shared memory pointer
                ptr_type = self._infer_type_from_ssa(ptr_ssa)
                if ptr_type == "smem" or "smem" in str(ptr_ssa):
                    ssa_val = self.mlir_generator.add_gpu_smem_load(ptr_ssa, offset_ssa)
                else:
                    ssa_val = self.mlir_generator.add_gpu_load(ptr_ssa, offset_ssa)

                self._track_ssa_type(
                    ssa_val, "f32"
                )  # Loads typically return f32 in GPU context
                return ssa_val
        elif func_name == "__load_from_ptr":
            # __load_from_ptr(ptr, offset)
            if len(node.args) >= 2:
                ptr_ssa = self.visit(node.args[0])
                offset_ssa = self.visit(node.args[1])
                ssa_val = self.mlir_generator.add_gpu_load(ptr_ssa, offset_ssa)
                self._track_ssa_type(
                    ssa_val, "f32"
                )  # Loads typically return f32 in GPU context
                return ssa_val
        elif func_name in ["store", "oven_lang_store"]:
            # store(value, ptr, offset)
            if len(node.args) >= 3:
                value_ssa = self.visit(node.args[0])
                ptr_ssa = self.visit(node.args[1])
                offset_ssa = self.visit(node.args[2])

                # Check if the pointer is a shared memory pointer
                ptr_type = self._infer_type_from_ssa(ptr_ssa)
                if ptr_type == "smem" or "smem" in str(ptr_ssa):
                    self.mlir_generator.add_gpu_smem_store(
                        value_ssa, ptr_ssa, offset_ssa
                    )
                else:
                    self.mlir_generator.add_gpu_store(value_ssa, ptr_ssa, offset_ssa)
                return value_ssa  # Return the stored value as SSA
        elif func_name in ["vload", "oven_lang_vload"]:
            # vload(ptr, offset, size)
            if len(node.args) >= 3:
                ptr_ssa = self.visit(node.args[0])
                offset_ssa = self.visit(node.args[1])
                # Size should be a constant integer
                size_node = node.args[2]
                if isinstance(size_node, ast.Constant):
                    size = size_node.value
                elif isinstance(size_node, ast.Num):  # Python < 3.8 compatibility
                    size = size_node.n
                else:
                    raise ValueError("vload size parameter must be a constant integer")

                ssa_val = self.mlir_generator.add_gpu_vload(ptr_ssa, offset_ssa, size)
                self._track_ssa_type(ssa_val, f"vector<{size}xf32>")
                return ssa_val
        elif func_name in ["vstore", "oven_lang_vstore"]:
            # vstore(vector, ptr, offset, size)
            if len(node.args) >= 4:
                vector_ssa = self.visit(node.args[0])
                ptr_ssa = self.visit(node.args[1])
                offset_ssa = self.visit(node.args[2])
                # Size should be a constant integer
                size_node = node.args[3]
                if isinstance(size_node, ast.Constant):
                    size = size_node.value
                elif isinstance(size_node, ast.Num):  # Python < 3.8 compatibility
                    size = size_node.n
                else:
                    raise ValueError("vstore size parameter must be a constant integer")

                self.mlir_generator.add_gpu_vstore(
                    vector_ssa, ptr_ssa, offset_ssa, size
                )
                return vector_ssa  # Return the vector value as SSA
        elif func_name == "__store_to_ptr":
            # __store_to_ptr(ptr, offset, value)
            if len(node.args) >= 3:
                ptr_ssa = self.visit(node.args[0])
                offset_ssa = self.visit(node.args[1])
                value_ssa = self.visit(node.args[2])
                self.mlir_generator.add_gpu_store(value_ssa, ptr_ssa, offset_ssa)
                return value_ssa  # Return the stored value as SSA
        elif func_name in ["smem", "oven_lang_smem"]:
            # smem() - allocate shared memory
            ssa_val = self.mlir_generator.add_gpu_smem()
            self._track_ssa_type(ssa_val, "smem")  # Track as shared memory
            return ssa_val
        elif func_name in ["barrier", "oven_lang_barrier"]:
            # barrier() - synchronization barrier
            self.mlir_generator.add_gpu_barrier()
            return None  # barrier doesn't return a value
        elif func_name in ["load_input_x", "oven_lang_load_input_x"]:
            # load_input_x(index) - load from input buffer x
            if len(node.args) >= 1:
                index_ssa = self.visit(node.args[0])
                ssa_val = self.mlir_generator.add_gpu_load_input_x(index_ssa)
                self._track_ssa_type(ssa_val, "f32")  # Input loads return f32
                return ssa_val
        elif func_name in ["store_output_x", "oven_lang_store_output_x"]:
            # store_output_x(value, index) - store to output buffer x
            if len(node.args) >= 2:
                value_ssa = self.visit(node.args[0])
                index_ssa = self.visit(node.args[1])
                self.mlir_generator.add_gpu_store_output_x(value_ssa, index_ssa)
                return value_ssa  # Return the stored value
        elif func_name in ["load_input_y", "oven_lang_load_input_y"]:
            # load_input_y(index) - load from input buffer y
            if len(node.args) >= 1:
                index_ssa = self.visit(node.args[0])
                ssa_val = self.mlir_generator.add_gpu_load_input_y(index_ssa)
                self._track_ssa_type(ssa_val, "f32")  # Input loads return f32
                return ssa_val
        elif func_name in ["store_output_y", "oven_lang_store_output_y"]:
            # store_output_y(value, index) - store to output buffer y
            if len(node.args) >= 2:
                value_ssa = self.visit(node.args[0])
                index_ssa = self.visit(node.args[1])
                self.mlir_generator.add_gpu_store_output_y(value_ssa, index_ssa)
                return value_ssa  # Return the stored value
        # Handle mathematical function calls
        elif func_name in ["exp", "oven_lang_exp"]:
            # exp(value)
            if len(node.args) >= 1:
                operand_ssa = self.visit(node.args[0])
                # Get operand type for vector support
                operand_type = self._infer_type_from_ssa(operand_ssa)
                if operand_type and "vector<" in str(operand_type):
                    result_ssa = self.mlir_generator.add_math_exp(
                        operand_ssa, operand_type
                    )
                    self._track_ssa_type(result_ssa, operand_type)
                else:
                    result_ssa = self.mlir_generator.add_math_exp(operand_ssa)
                    self._track_ssa_type(result_ssa, "f32")
                return result_ssa
        elif func_name in ["exp2", "oven_lang_exp2"]:
            # exp2(value)
            if len(node.args) >= 1:
                operand_ssa = self.visit(node.args[0])
                # Get operand type for vector support
                operand_type = self._infer_type_from_ssa(operand_ssa)
                if operand_type and "vector<" in str(operand_type):
                    result_ssa = self.mlir_generator.add_math_exp2(
                        operand_ssa, operand_type
                    )
                    self._track_ssa_type(result_ssa, operand_type)
                else:
                    result_ssa = self.mlir_generator.add_math_exp2(operand_ssa)
                    self._track_ssa_type(result_ssa, "f32")
                return result_ssa
        elif func_name in ["sigmoid", "oven_lang_sigmoid"]:
            # sigmoid(value)
            if len(node.args) >= 1:
                operand_ssa = self.visit(node.args[0])
                # Get operand type for vector support
                operand_type = self._infer_type_from_ssa(operand_ssa)
                if operand_type and "vector<" in str(operand_type):
                    result_ssa = self.mlir_generator.add_oven_sigmoid(
                        operand_ssa, operand_type
                    )
                    self._track_ssa_type(result_ssa, operand_type)
                else:
                    result_ssa = self.mlir_generator.add_oven_sigmoid(operand_ssa)
                    self._track_ssa_type(result_ssa, "f32")
                return result_ssa
        elif func_name in ["sin", "oven_lang_sin"]:
            # sin(value)
            if len(node.args) >= 1:
                operand_ssa = self.visit(node.args[0])
                # Get operand type for vector support
                operand_type = self._infer_type_from_ssa(operand_ssa)
                if operand_type and "vector<" in str(operand_type):
                    result_ssa = self.mlir_generator.add_math_sin(
                        operand_ssa, operand_type
                    )
                    self._track_ssa_type(result_ssa, operand_type)
                else:
                    result_ssa = self.mlir_generator.add_math_sin(operand_ssa)
                    self._track_ssa_type(result_ssa, "f32")
                return result_ssa
        elif func_name in ["cos", "oven_lang_cos"]:
            # cos(value)
            if len(node.args) >= 1:
                operand_ssa = self.visit(node.args[0])
                # Get operand type for vector support
                operand_type = self._infer_type_from_ssa(operand_ssa)
                if operand_type and "vector<" in str(operand_type):
                    result_ssa = self.mlir_generator.add_math_cos(
                        operand_ssa, operand_type
                    )
                    self._track_ssa_type(result_ssa, operand_type)
                else:
                    result_ssa = self.mlir_generator.add_math_cos(operand_ssa)
                    self._track_ssa_type(result_ssa, "f32")
                return result_ssa
        elif func_name in ["tan", "oven_lang_tan"]:
            # tan(value)
            if len(node.args) >= 1:
                operand_ssa = self.visit(node.args[0])
                # Get operand type for vector support
                operand_type = self._infer_type_from_ssa(operand_ssa)
                if operand_type and "vector<" in str(operand_type):
                    result_ssa = self.mlir_generator.add_math_tan(
                        operand_ssa, operand_type
                    )
                    self._track_ssa_type(result_ssa, operand_type)
                else:
                    result_ssa = self.mlir_generator.add_math_tan(operand_ssa)
                    self._track_ssa_type(result_ssa, "f32")
                return result_ssa
        elif func_name in ["sqrt", "oven_lang_sqrt"]:
            # sqrt(value)
            if len(node.args) >= 1:
                operand_ssa = self.visit(node.args[0])
                # Get operand type for vector support
                operand_type = self._infer_type_from_ssa(operand_ssa)
                if operand_type and "vector<" in str(operand_type):
                    result_ssa = self.mlir_generator.add_math_sqrt(
                        operand_ssa, operand_type
                    )
                    self._track_ssa_type(result_ssa, operand_type)
                else:
                    result_ssa = self.mlir_generator.add_math_sqrt(operand_ssa)
                    self._track_ssa_type(result_ssa, "f32")
                return result_ssa
        elif func_name in ["log", "oven_lang_log"]:
            # log(value)
            if len(node.args) >= 1:
                operand_ssa = self.visit(node.args[0])
                # Get operand type for vector support
                operand_type = self._infer_type_from_ssa(operand_ssa)
                if operand_type and "vector<" in str(operand_type):
                    result_ssa = self.mlir_generator.add_math_log(
                        operand_ssa, operand_type
                    )
                    self._track_ssa_type(result_ssa, operand_type)
                else:
                    result_ssa = self.mlir_generator.add_math_log(operand_ssa)
                    self._track_ssa_type(result_ssa, "f32")
                return result_ssa
        elif func_name in ["log2", "oven_lang_log2"]:
            # log2(value)
            if len(node.args) >= 1:
                operand_ssa = self.visit(node.args[0])
                # Get operand type for vector support
                operand_type = self._infer_type_from_ssa(operand_ssa)
                if operand_type and "vector<" in str(operand_type):
                    result_ssa = self.mlir_generator.add_math_log2(
                        operand_ssa, operand_type
                    )
                    self._track_ssa_type(result_ssa, operand_type)
                else:
                    result_ssa = self.mlir_generator.add_math_log2(operand_ssa)
                    self._track_ssa_type(result_ssa, "f32")
                return result_ssa
        elif func_name in ["abs", "oven_lang_abs"]:
            # abs(value)
            if len(node.args) >= 1:
                operand_ssa = self.visit(node.args[0])
                # Get operand type for vector support
                operand_type = self._infer_type_from_ssa(operand_ssa)
                if operand_type and "vector<" in str(operand_type):
                    result_ssa = self.mlir_generator.add_math_absf(
                        operand_ssa, operand_type
                    )
                    self._track_ssa_type(result_ssa, operand_type)
                else:
                    result_ssa = self.mlir_generator.add_math_absf(operand_ssa)
                    self._track_ssa_type(result_ssa, "f32")
                return result_ssa
        elif func_name in ["ceil", "oven_lang_ceil"]:
            # ceil(value)
            if len(node.args) >= 1:
                operand_ssa = self.visit(node.args[0])
                # Get operand type for vector support
                operand_type = self._infer_type_from_ssa(operand_ssa)
                if operand_type and "vector<" in str(operand_type):
                    result_ssa = self.mlir_generator.add_math_ceil(
                        operand_ssa, operand_type
                    )
                    self._track_ssa_type(result_ssa, operand_type)
                else:
                    result_ssa = self.mlir_generator.add_math_ceil(operand_ssa)
                    self._track_ssa_type(result_ssa, "f32")
                return result_ssa
        elif func_name in ["floor", "oven_lang_floor"]:
            # floor(value)
            if len(node.args) >= 1:
                operand_ssa = self.visit(node.args[0])
                # Get operand type for vector support
                operand_type = self._infer_type_from_ssa(operand_ssa)
                if operand_type and "vector<" in str(operand_type):
                    result_ssa = self.mlir_generator.add_math_floor(
                        operand_ssa, operand_type
                    )
                    self._track_ssa_type(result_ssa, operand_type)
                else:
                    result_ssa = self.mlir_generator.add_math_floor(operand_ssa)
                    self._track_ssa_type(result_ssa, "f32")
                return result_ssa
        elif func_name in ["rsqrt", "oven_lang_rsqrt"]:
            # rsqrt(value)
            if len(node.args) >= 1:
                operand_ssa = self.visit(node.args[0])
                # Get operand type for vector support
                operand_type = self._infer_type_from_ssa(operand_ssa)
                if operand_type and "vector<" in str(operand_type):
                    result_ssa = self.mlir_generator.add_math_rsqrt(
                        operand_ssa, operand_type
                    )
                    self._track_ssa_type(result_ssa, operand_type)
                else:
                    result_ssa = self.mlir_generator.add_math_rsqrt(operand_ssa)
                    self._track_ssa_type(result_ssa, "f32")
                return result_ssa
        # Handle arithmetic function calls
        elif func_name in ["muli", "oven_lang_muli"]:
            # muli(a, b)
            if len(node.args) >= 2:
                lhs_ssa = self.visit(node.args[0])
                rhs_ssa = self.visit(node.args[1])
                return self.mlir_generator.add_arith_muli(lhs_ssa, rhs_ssa)
        elif func_name in ["addi", "oven_lang_addi"]:
            # addi(a, b)
            if len(node.args) >= 2:
                lhs_ssa = self.visit(node.args[0])
                rhs_ssa = self.visit(node.args[1])
                return self.mlir_generator.add_arith_addi(lhs_ssa, rhs_ssa)
        elif func_name in ["mulf", "oven_lang_mulf"]:
            # mulf(a, b)
            if len(node.args) >= 2:
                lhs_ssa = self.visit(node.args[0])
                rhs_ssa = self.visit(node.args[1])
                return self.mlir_generator.add_arith_mulf(lhs_ssa, rhs_ssa)
        elif func_name in ["addf", "oven_lang_addf"]:
            # addf(a, b)
            if len(node.args) >= 2:
                lhs_ssa = self.visit(node.args[0])
                rhs_ssa = self.visit(node.args[1])
                return self.mlir_generator.add_arith_addf(lhs_ssa, rhs_ssa)
        # Handle type conversion function calls
        elif func_name in ["index_cast", "oven_lang_index_cast"]:
            # index_cast(value, from_type, to_type)
            if len(node.args) >= 3:
                value_ssa = self.visit(node.args[0])
                from_type = (
                    str(node.args[1].s)
                    if isinstance(node.args[1], ast.Str)
                    else "index"
                )
                to_type = (
                    str(node.args[2].s) if isinstance(node.args[2], ast.Str) else "i32"
                )
                return self.mlir_generator.add_arith_index_cast(
                    value_ssa, from_type, to_type
                )
        # Handle constant function calls
        elif func_name in ["constant", "oven_lang_constant"]:
            # constant(value, type)
            if len(node.args) >= 2:
                if isinstance(node.args[0], ast.Constant):
                    value = node.args[0].value
                    if isinstance(value, int):
                        if (
                            isinstance(node.args[1], ast.Str)
                            and node.args[1].s == "index"
                        ):
                            return self.mlir_generator.add_constant_index(value)
                        else:
                            return self.mlir_generator.add_constant_int(value)
                    elif isinstance(value, float):
                        return self.mlir_generator.add_constant_float(value)

        # Visit arguments for regular function calls
        arg_ssa_values = []
        for arg in node.args:
            arg_ssa_values.append(self.visit(arg))

        return self.mlir_generator.add_function_call(func_name, arg_ssa_values)

    def visit_BinOp(self, node: ast.BinOp) -> str:
        """Visit a binary operation (e.g., a + b, a * b)."""
        left_ssa = self.visit(node.left)
        right_ssa = self.visit(node.right)

        # Determine operation type based on operands
        # For now, we'll use simple heuristics - can be enhanced later

        if isinstance(node.op, ast.Add):
            # Addition: + -> addi or addf
            return self._generate_arithmetic_op("add", left_ssa, right_ssa)
        elif isinstance(node.op, ast.Mult):
            # Multiplication: * -> muli or mulf
            return self._generate_arithmetic_op("mul", left_ssa, right_ssa)
        elif isinstance(node.op, ast.Sub):
            # Subtraction: - -> subi or subf
            return self._generate_arithmetic_op("sub", left_ssa, right_ssa)
        elif isinstance(node.op, ast.Div):
            # Division: / -> divf (usually floating point)
            return self._generate_arithmetic_op(
                "div", left_ssa, right_ssa, force_float=True
            )
        else:
            # Fallback for unsupported operations
            return self.mlir_generator.add_constant_int(0)

    def _generate_arithmetic_op(
        self, op: str, left_ssa: str, right_ssa: str, force_float: bool = False
    ) -> str:
        """Generate appropriate arithmetic operation based on operand types."""

        # Get types of the operands
        left_type = self._infer_type_from_ssa(left_ssa)
        right_type = self._infer_type_from_ssa(right_ssa)

        # Check if either operand is a vector type
        is_vector_op = (left_type and "vector<" in str(left_type)) or (
            right_type and "vector<" in str(right_type)
        )

        if is_vector_op:
            # Extract vector type info
            vector_type = (
                left_type if left_type and "vector<" in str(left_type) else right_type
            )

            # Generate vector arithmetic operation
            if op == "add":
                ssa_val = self.mlir_generator.add_arith_addf(
                    left_ssa, right_ssa, vector_type
                )
            elif op == "mul":
                ssa_val = self.mlir_generator.add_arith_mulf(
                    left_ssa, right_ssa, vector_type
                )
            elif op == "sub":
                ssa_val = self.mlir_generator.add_arith_subf(
                    left_ssa, right_ssa, vector_type
                )
            elif op == "div":
                ssa_val = self.mlir_generator.add_arith_divf(
                    left_ssa, right_ssa, vector_type
                )
            else:
                ssa_val = self.mlir_generator.add_constant_int(0)

            # Track the result type as vector
            self._track_ssa_type(ssa_val, vector_type)
            return ssa_val

        # Determine if we should use float operations
        if force_float:
            use_float = True
        elif left_type == "f32" or right_type == "f32":
            # If either operand is float, use float operation
            use_float = True
        elif left_type == "!llvm.ptr" or right_type == "!llvm.ptr":
            # Pointer arithmetic should use integer operations
            use_float = False
        elif left_type == "i32" and right_type == "i32":
            # Both operands are integers, use integer operations
            use_float = False
        elif left_type == "index" or right_type == "index":
            # Index arithmetic uses integer operations
            use_float = False
        else:
            # Enhanced heuristic: analyze the context as fallback
            is_gpu_context = hasattr(self, "_current_is_gpu") and self._current_is_gpu
            is_math_context = (
                hasattr(self, "_current_is_math") and self._current_is_math
            )

            if is_math_context:
                use_float = True
            elif is_gpu_context:
                # In GPU context, default to float for data operations only when types are unknown
                use_float = True
            else:
                # For regular functions, use integer operations unless types suggest otherwise
                use_float = False

        # Generate the appropriate operation
        if use_float:
            result_type = "f32"
            if op == "add":
                ssa_val = self.mlir_generator.add_arith_addf(left_ssa, right_ssa)
            elif op == "mul":
                ssa_val = self.mlir_generator.add_arith_mulf(left_ssa, right_ssa)
            elif op == "sub":
                ssa_val = self.mlir_generator.add_arith_subf(left_ssa, right_ssa)
            elif op == "div":
                ssa_val = self.mlir_generator.add_arith_divf(left_ssa, right_ssa)
            else:
                ssa_val = self.mlir_generator.add_constant_int(0)
        else:
            result_type = "i32"
            if op == "add":
                ssa_val = self.mlir_generator.add_arith_addi(left_ssa, right_ssa)
            elif op == "mul":
                ssa_val = self.mlir_generator.add_arith_muli(left_ssa, right_ssa)
            elif op == "sub":
                ssa_val = self.mlir_generator.add_arith_subi(left_ssa, right_ssa)
            elif op == "div":
                # Division is typically floating point even for integers
                result_type = "f32"
                ssa_val = self.mlir_generator.add_arith_divf(left_ssa, right_ssa, "f32")
            else:
                ssa_val = self.mlir_generator.add_constant_int(0)

        # Track the result type
        self._track_ssa_type(ssa_val, result_type)
        return ssa_val

    def visit_Expr(self, node: ast.Expr) -> Any:
        """Visit an expression statement."""
        self.visit(node.value)

    def generic_visit(self, node: ast.AST) -> Any:
        """Generic visit method for unsupported nodes."""
        # For unsupported nodes, just visit children
        super().generic_visit(node)
