__all__ = [
    "apply_manual_api_refinement"
]

from typing import Sequence, Callable

from .nodes import (NamespaceNode, FunctionNode, OptionalTypeNode,
                    ClassProperty, PrimitiveTypeNode)
from .ast_utils import find_function_node, SymbolName


def apply_manual_api_refinement(root: NamespaceNode) -> None:
    export_matrix_type_constants(root)
    # Export OpenCV exception class
    builtin_exception = root.add_class("Exception")
    builtin_exception.is_exported = False
    root.add_class("error", (builtin_exception, ), ERROR_CLASS_PROPERTIES)
    for symbol_name, refine_symbol in NODES_TO_REFINE.items():
        refine_symbol(root, symbol_name)


def export_matrix_type_constants(root: NamespaceNode) -> None:
    MAX_PREDEFINED_CHANNELS = 4

    depth_names = ("CV_8U", "CV_8S", "CV_16U", "CV_16S", "CV_32S",
                   "CV_32F", "CV_64F", "CV_16F")
    for depth_value, depth_name in enumerate(depth_names):
        # Export depth constants
        root.add_constant(depth_name, str(depth_value))
        # Export predefined types
        for c in range(MAX_PREDEFINED_CHANNELS):
            root.add_constant(f"{depth_name}C{c + 1}",
                              f"{depth_value + 8 * c}")
        # Export type creation function
        root.add_function(
            f"{depth_name}C",
            (FunctionNode.Arg("channels", PrimitiveTypeNode.int_()), ),
            FunctionNode.RetType(PrimitiveTypeNode.int_())
        )
    # Export CV_MAKETYPE
    root.add_function(
        "CV_MAKETYPE",
        (FunctionNode.Arg("depth", PrimitiveTypeNode.int_()),
         FunctionNode.Arg("channels", PrimitiveTypeNode.int_())),
        FunctionNode.RetType(PrimitiveTypeNode.int_())
    )


def make_optional_arg(arg_name: str) -> Callable[[NamespaceNode, SymbolName], None]:
    def _make_optional_arg(root_node: NamespaceNode,
                           function_symbol_name: SymbolName) -> None:
        function = find_function_node(root_node, function_symbol_name)
        for overload in function.overloads:
            arg_idx = _find_argument_index(overload.arguments, arg_name)
            # Avoid multiplying optional qualification
            if isinstance(overload.arguments[arg_idx].type_node, OptionalTypeNode):
                continue

            overload.arguments[arg_idx].type_node = OptionalTypeNode(
                overload.arguments[arg_idx].type_node
            )

    return _make_optional_arg


def _find_argument_index(arguments: Sequence[FunctionNode.Arg], name: str) -> int:
    for i, arg in enumerate(arguments):
        if arg.name == name:
            return i
    raise RuntimeError(
        f"Failed to find argument with name: '{name}' in {arguments}"
    )


NODES_TO_REFINE = {
    SymbolName(("cv", ), (), "resize"): make_optional_arg("dsize"),
    SymbolName(("cv", ), (), "calcHist"): make_optional_arg("mask"),
}
ERROR_CLASS_PROPERTIES = (
    ClassProperty("code", PrimitiveTypeNode.int_(), False),
    ClassProperty("err", PrimitiveTypeNode.str_(), False),
    ClassProperty("file", PrimitiveTypeNode.str_(), False),
    ClassProperty("func", PrimitiveTypeNode.str_(), False),
    ClassProperty("line", PrimitiveTypeNode.int_(), False),
    ClassProperty("msg", PrimitiveTypeNode.str_(), False),
)
