from typing import Optional, Tuple

from .node import ASTNode, ASTNodeType


class ConstantNode(ASTNode):
    """Represents C++ constant that is also a constant in Python.
    """
    def __init__(self, name: str, value: str,
                 parent: Optional[ASTNode] = None,
                 export_name: Optional[str] = None) -> None:
        super().__init__(name, parent, export_name)
        self.value = value
        self._value_type = "int"

    @property
    def children_types(self) -> Tuple[ASTNodeType, ...]:
        return ()

    @property
    def node_type(self) -> ASTNodeType:
        return ASTNodeType.Constant

    @property
    def value_type(self) -> str:
        return self._value_type

    def __str__(self) -> str:
        return "Constant('{}' exported as '{}': {})".format(
            self.name, self.export_name, self.value
        )
