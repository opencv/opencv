import itertools
import weakref
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Type

from .class_node import ClassNode, ClassProperty
from .constant_node import ConstantNode
from .enumeration_node import EnumerationNode
from .function_node import FunctionNode
from .node import ASTNode, ASTNodeType
from .type_node import TypeResolutionError


class NamespaceNode(ASTNode):
    """Represents C++ namespace that treated as module in Python.

    NamespaceNode can have other namespaces, classes, functions, enumerations
    and global constants as its children nodes.
    """
    def __init__(self, name: str, parent: Optional[ASTNode] = None,
                 export_name: Optional[str] = None) -> None:
        super().__init__(name, parent, export_name)
        self.reexported_submodules: List[str] = []
        """List of reexported submodules"""

        self.reexported_submodules_symbols: Dict[str, List[str]] = defaultdict(list)
        """Mapping between submodules export names and their symbols re-exported
        in this module"""


    @property
    def node_type(self) -> ASTNodeType:
        return ASTNodeType.Namespace

    @property
    def children_types(self) -> Tuple[Type[ASTNode], ...]:
        return (NamespaceNode, ClassNode, FunctionNode,
                EnumerationNode, ConstantNode)

    @property
    def namespaces(self) -> Dict[str, "NamespaceNode"]:
        return self._children[NamespaceNode]

    @property
    def classes(self) -> Dict[str, ClassNode]:
        return self._children[ClassNode]

    @property
    def functions(self) -> Dict[str, FunctionNode]:
        return self._children[FunctionNode]

    @property
    def enumerations(self) -> Dict[str, EnumerationNode]:
        return self._children[EnumerationNode]

    @property
    def constants(self) -> Dict[str, ConstantNode]:
        return self._children[ConstantNode]

    def add_namespace(self, name: str) -> "NamespaceNode":
        return self._add_child(NamespaceNode, name)

    def add_class(self, name: str,
                  bases: Sequence["weakref.ProxyType[ClassNode]"] = (),
                  properties: Sequence[ClassProperty] = ()) -> "ClassNode":
        return self._add_child(ClassNode, name, bases=bases,
                               properties=properties)

    def add_function(self, name: str, arguments: Sequence[FunctionNode.Arg] = (),
                     return_type: Optional[FunctionNode.RetType] = None) -> FunctionNode:
        return self._add_child(FunctionNode, name, arguments=arguments,
                               return_type=return_type)

    def add_enumeration(self, name: str) -> EnumerationNode:
        return self._add_child(EnumerationNode, name)

    def add_constant(self, name: str, value: str) -> ConstantNode:
        return self._add_child(ConstantNode, name, value=value)

    def resolve_type_nodes(self, root: Optional[ASTNode] = None) -> None:
        """Resolves type nodes for all children nodes in 2 steps:
            1. Resolve against `self` as a tree root
            2. Resolve against `root` as a tree root
        Type resolution errors are postponed until all children nodes are
        examined.

        Args:
            root (Optional[ASTNode], optional): Root of the AST sub-tree.
                Defaults to None.
        """
        errors = []
        for child in itertools.chain(self.functions.values(),
                                     self.classes.values(),
                                     self.namespaces.values()):
            try:
                try:
                    child.resolve_type_nodes(self)  # type: ignore
                except TypeResolutionError:
                    if root is not None:
                        child.resolve_type_nodes(root)  # type: ignore
                    else:
                        raise
            except TypeResolutionError as e:
                errors.append(str(e))
        if len(errors) > 0:
            raise TypeResolutionError(
                'Failed to resolve "{}" namespace against "{}". '
                'Errors: {}'.format(
                    self.full_export_name,
                    root if root is None else root.full_export_name,
                    errors
                )
            )
