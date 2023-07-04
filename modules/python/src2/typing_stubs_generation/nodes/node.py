import abc
import enum
import itertools
from typing import (Iterator, Type, TypeVar, Dict,
                    Optional, Tuple, DefaultDict)
from collections import defaultdict

import weakref


ASTNodeSubtype = TypeVar("ASTNodeSubtype", bound="ASTNode")
NodeType = Type["ASTNode"]
NameToNode = Dict[str, ASTNodeSubtype]


class ASTNodeType(enum.Enum):
    Namespace = enum.auto()
    Class = enum.auto()
    Function = enum.auto()
    Enumeration = enum.auto()
    Constant = enum.auto()


class ASTNode:
    """Represents an element of the Abstract Syntax Tree produced by parsing
    public C++ headers.

    NOTE: Every node manages a lifetime of its children nodes. Children nodes
    contain only weak references to their direct parents, so there are no
    circular dependencies.
    """

    def __init__(self, name: str, parent: Optional["ASTNode"] = None,
                 export_name: Optional[str] = None) -> None:
        """ASTNode initializer

        Args:
            name (str): name of the node, should be unique inside enclosing
                context (There can't be 2 classes with the same name defined
                in the same namespace).
            parent (ASTNode, optional): parent node expressing node context.
                None corresponds to globally defined object e.g. root namespace
                or function without namespace. Defaults to None.
            export_name (str, optional): export name of the node used to resolve
                issues in languages without proper overload resolution and
                provide more meaningful naming. Defaults to None.
        """

        FORBIDDEN_SYMBOLS = ";,*&#/|\\@!()[]^% "
        for forbidden_symbol in FORBIDDEN_SYMBOLS:
            assert forbidden_symbol not in name, \
                "Invalid node identifier '{}' - contains 1 or more "\
                "forbidden symbols: ({})".format(name, FORBIDDEN_SYMBOLS)

        assert ":" not in name, \
            "Name '{}' contains C++ scope symbols (':'). Convert the name to "\
            "Python style and create appropriate parent nodes".format(name)

        assert "." not in name, \
            "Trying to create a node with '.' symbols in its name ({}). " \
            "Dots are supposed to be a scope delimiters, so create all nodes in ('{}') " \
            "and add '{}' as a last child node".format(
                name,
                "->".join(name.split('.')[:-1]),
                name.rsplit('.', maxsplit=1)[-1]
            )

        self.__name = name
        self.export_name = name if export_name is None else export_name
        self._parent: Optional["ASTNode"] = None
        self.parent = parent
        self.is_exported = True
        self._children: DefaultDict[NodeType, NameToNode] = defaultdict(dict)

    def __str__(self) -> str:
        return "{}('{}' exported as '{}')".format(
            type(self).__name__.replace("Node", ""), self.name, self.export_name
        )

    def __repr__(self) -> str:
        return str(self)

    @abc.abstractproperty
    def children_types(self) -> Tuple[Type["ASTNode"], ...]:
        """Set of ASTNode types that are allowed to be children of this node

        Returns:
            Tuple[Type[ASTNode], ...]: Types of children nodes
        """
        pass

    @abc.abstractproperty
    def node_type(self) -> ASTNodeType:
        """Type of the ASTNode that can be used to distinguish nodes without
        importing all subclasses of ASTNode

        Returns:
            ASTNodeType: Current node type
        """
        pass

    @property
    def name(self) -> str:
        return self.__name

    @property
    def native_name(self) -> str:
        return self.full_name.replace(".", "::")

    @property
    def full_name(self) -> str:
        return self._construct_full_name("name")

    @property
    def full_export_name(self) -> str:
        return self._construct_full_name("export_name")

    @property
    def parent(self) -> Optional["ASTNode"]:
        return self._parent

    @parent.setter
    def parent(self, value: Optional["ASTNode"]) -> None:
        assert value is None or isinstance(value, ASTNode), \
            "ASTNode.parent should be None or another ASTNode, " \
            "but got: {}".format(type(value))

        if value is not None:
            value.__check_child_before_add(type(self), self.name)

        # Detach from previous parent
        if self._parent is not None:
            self._parent._children[type(self)].pop(self.name)

        if value is None:
            self._parent = None
            return

        # Set a weak reference to a new parent and add self to its children
        self._parent = weakref.proxy(value)
        value._children[type(self)][self.name] = self

    def __check_child_before_add(self, child_type: Type[ASTNodeSubtype],
                                 name: str) -> None:
        assert len(self.children_types) > 0, \
            "Trying to add child node '{}::{}' to node '{}::{}' " \
            "that can't have children nodes".format(child_type.__name__, name,
                                                    type(self).__name__,
                                                    self.name)

        assert child_type in self.children_types, \
            "Trying to add child node '{}::{}' to node '{}::{}' " \
            "that supports only ({}) as its children types".format(
                child_type.__name__, name, type(self).__name__, self.name,
                ",".join(t.__name__ for t in self.children_types)
            )

        if self._find_child(child_type, name) is not None:
            raise ValueError(
                "Node '{}::{}' already has a child '{}::{}'".format(
                    type(self).__name__, self.name, child_type.__name__, name
                )
            )

    def _add_child(self, child_type: Type[ASTNodeSubtype], name: str,
                   **kwargs) -> ASTNodeSubtype:
        """Creates a child of the node with the given type and performs common
        validation checks:
        - Node can have children of the provided type
        - Node doesn't have child with the same name

        NOTE: Shouldn't be used directly by a user.

        Args:
            child_type (Type[ASTNodeSubtype]): Type of the child to create.
            name (str): Name of the child.
            **kwargs: Extra keyword arguments supplied to child_type.__init__
                method.

        Returns:
            ASTNodeSubtype: Created ASTNode
        """
        self.__check_child_before_add(child_type, name)
        return child_type(name, parent=self, **kwargs)

    def _find_child(self, child_type: Type[ASTNodeSubtype],
                    name: str) -> Optional[ASTNodeSubtype]:
        """Looks for child node with the given type and name.

        Args:
            child_type (Type[ASTNodeSubtype]): Type of the child node.
            name (str): Name of the child node.

        Returns:
            Optional[ASTNodeSubtype]: child node if it can be found, None
                otherwise.
        """
        if child_type not in self._children:
            return None
        return self._children[child_type].get(name, None)

    def _construct_full_name(self, property_name: str) -> str:
        """Traverses nodes hierarchy upright to the root node and constructs a
        full name of the node using original or export names depending on the
        provided `property_name` argument.

        Args:
            property_name (str): Name of the property to quire from node to get
                its name. Should be `name` or `export_name`.

        Returns:
            str: full node name where each node part is divided with a dot.
        """
        def get_name(node: ASTNode) -> str:
            return getattr(node, property_name)

        assert property_name in ('name', 'export_name'), 'Invalid name property'

        name_parts = [get_name(self), ]
        parent = self.parent
        while parent is not None:
            name_parts.append(get_name(parent))
            parent = parent.parent
        return ".".join(reversed(name_parts))

    def __iter__(self) -> Iterator["ASTNode"]:
        return iter(itertools.chain.from_iterable(
            node
            # Iterate over mapping between node type and nodes dict
            for children_nodes in self._children.values()
            # Iterate over mapping between node name and node
            for node in children_nodes.values()
        ))
