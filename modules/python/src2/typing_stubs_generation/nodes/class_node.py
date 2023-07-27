from typing import Type, Sequence, NamedTuple, Optional, Tuple, Dict
import itertools

import weakref

from .node import ASTNode, ASTNodeType

from .function_node import FunctionNode
from .enumeration_node import EnumerationNode
from .constant_node import ConstantNode

from .type_node import TypeNode, TypeResolutionError


class ClassProperty(NamedTuple):
    name: str
    type_node: TypeNode
    is_readonly: bool

    @property
    def typename(self) -> str:
        return self.type_node.full_typename

    def resolve_type_nodes(self, root: ASTNode) -> None:
        try:
            self.type_node.resolve(root)
        except TypeResolutionError as e:
            raise TypeResolutionError(
                'Failed to resolve "{}" property'.format(self.name)
            ) from e

    def relative_typename(self, full_node_name: str) -> str:
        """Typename relative to the passed AST node name.

        Args:
            full_node_name (str): Full export name of the AST node

        Returns:
            str: typename relative to the passed AST node name
        """
        return self.type_node.relative_typename(full_node_name)


class ClassNode(ASTNode):
    """Represents a C++ class that is also a class in Python.

    ClassNode can have functions (methods), enumerations, constants and other
    classes as its children nodes.

    Class properties are not treated as a part of AST for simplicity and have
    extra handling if required.
    """
    def __init__(self, name: str, parent: Optional[ASTNode] = None,
                 export_name: Optional[str] = None,
                 bases: Sequence["weakref.ProxyType[ClassNode]"] = (),
                 properties: Sequence[ClassProperty] = ()) -> None:
        super().__init__(name, parent, export_name)
        self.bases = list(bases)
        self.properties = properties

    @property
    def weight(self) -> int:
        return 1 + sum(base.weight for base in self.bases)

    @property
    def children_types(self) -> Tuple[ASTNodeType, ...]:
        return (ASTNodeType.Class, ASTNodeType.Function,
                ASTNodeType.Enumeration, ASTNodeType.Constant)

    @property
    def node_type(self) -> ASTNodeType:
        return ASTNodeType.Class

    @property
    def classes(self) -> Dict[str, "ClassNode"]:
        return self._children[ASTNodeType.Class]

    @property
    def functions(self) -> Dict[str, FunctionNode]:
        return self._children[ASTNodeType.Function]

    @property
    def enumerations(self) -> Dict[str, EnumerationNode]:
        return self._children[ASTNodeType.Enumeration]

    @property
    def constants(self) -> Dict[str, ConstantNode]:
        return self._children[ASTNodeType.Constant]

    def add_class(self, name: str,
                  bases: Sequence["weakref.ProxyType[ClassNode]"] = (),
                  properties: Sequence[ClassProperty] = ()) -> "ClassNode":
        return self._add_child(ClassNode, name, bases=bases,
                               properties=properties)

    def add_function(self, name: str, arguments: Sequence[FunctionNode.Arg] = (),
                     return_type: Optional[FunctionNode.RetType] = None,
                     is_static: bool = False) -> FunctionNode:
        """Adds function as a child node of a class.

        Function is classified in 3 categories:
            1. Instance method.
               If function is an instance method then `self` argument is
               inserted at the beginning of its arguments list.

            2. Class method (or factory method)
               If `is_static` flag is `True` and typename of the function
               return type matches name of the class then function is treated
               as class method.

               If function is a class method then `cls` argument is inserted
               at the beginning of its arguments list.

            3. Static method

        Args:
            name (str): Name of the function.
            arguments (Sequence[FunctionNode.Arg], optional): Function arguments.
                Defaults to ().
            return_type (Optional[FunctionNode.RetType], optional): Function
                return type. Defaults to None.
            is_static (bool, optional): Flag whenever function is static or not.
                Defaults to False.

        Returns:
            FunctionNode: created function node.
        """

        arguments = list(arguments)
        if return_type is not None:
            is_classmethod = return_type.typename == self.name
        else:
            is_classmethod = False
        if not is_static:
            arguments.insert(0, FunctionNode.Arg("self"))
        elif is_classmethod:
            is_static = False
            arguments.insert(0, FunctionNode.Arg("cls"))
        return self._add_child(FunctionNode, name, arguments=arguments,
                               return_type=return_type, is_static=is_static,
                               is_classmethod=is_classmethod)

    def add_enumeration(self, name: str) -> EnumerationNode:
        return self._add_child(EnumerationNode, name)

    def add_constant(self, name: str, value: str) -> ConstantNode:
        return self._add_child(ConstantNode, name, value=value)

    def add_base(self, base_class_node: "ClassNode") -> None:
        self.bases.append(weakref.proxy(base_class_node))

    def resolve_type_nodes(self, root: ASTNode) -> None:
        """Resolves type nodes for all inner-classes, methods and properties
        in 2 steps:
            1. Resolve against `self` as a tree root
            2. Resolve against `root` as a tree root
        Type resolution errors are postponed until all children nodes are
        examined.

        Args:
            root (Optional[ASTNode], optional): Root of the AST sub-tree.
                Defaults to None.
        """

        errors = []
        for child in itertools.chain(self.properties,
                                     self.functions.values(),
                                     self.classes.values()):
            try:
                try:
                    # Give priority to narrowest scope (class-level scope in this case)
                    child.resolve_type_nodes(self)  # type: ignore
                except TypeResolutionError:
                    child.resolve_type_nodes(root)  # type: ignore
            except TypeResolutionError as e:
                errors.append(str(e))
        if len(errors) > 0:
            raise TypeResolutionError(
                'Failed to resolve "{}" class against "{}". Errors: {}'.format(
                    self.full_export_name, root.full_export_name, errors
                )
            )


class ProtocolClassNode(ClassNode):
    def __init__(self, name: str, parent: Optional[ASTNode] = None,
                 export_name: Optional[str] = None,
                 properties: Sequence[ClassProperty] = ()) -> None:
        super().__init__(name, parent, export_name, bases=(),
                         properties=properties)
