__all__ = [
    "apply_manual_api_refinement"
]

from typing import Sequence, Callable
from .nodes import NamespaceNode, FunctionNode, OptionalTypeNode
from .ast_utils import find_function_node, SymbolName


def apply_manual_api_refinement(root: NamespaceNode) -> None:
    # Export OpenCV exception class
    builtin_exception = root.add_class("Exception")
    builtin_exception.is_exported = False
    root.add_class("error", (builtin_exception, ))
    for symbol_name, refine_symbol in NODES_TO_REFINE.items():
        refine_symbol(root, symbol_name)


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
