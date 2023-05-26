from typing import Type, Tuple, Optional, Dict

from .node import ASTNode, ASTNodeType

from .constant_node import ConstantNode


class EnumerationNode(ASTNode):
    """Represents C++ enumeration that treated as named set of constants in
    Python.

    EnumerationNode can have only constants as its children nodes.
    """
    def __init__(self, name: str, is_scoped: bool = False,
                 parent: Optional[ASTNode] = None,
                 export_name: Optional[str] = None) -> None:
        super().__init__(name, parent, export_name)
        self.is_scoped = is_scoped

    @property
    def children_types(self) -> Tuple[Type[ASTNode], ...]:
        return (ConstantNode, )

    @property
    def node_type(self) -> ASTNodeType:
        return ASTNodeType.Enumeration

    @property
    def constants(self) -> Dict[str, ConstantNode]:
        return self._children[ConstantNode]

    def add_constant(self, name: str, value: str) -> ConstantNode:
        return self._add_child(ConstantNode, name, value=value)
