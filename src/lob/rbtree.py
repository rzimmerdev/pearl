class Node:
    def __init__(self, key, color='red'):
        self.key = key
        self.left = None
        self.right = None
        self.parent = None
        self.color = color  # red by default


class RedBlackTree:
    def __init__(self):
        self.root = None

    def insert(self, key):
        node = Node(key)
        if self.root is None:
            self.root = node
        else:
            self._insert_recursive(self.root, node)
        self._fix_insert(node)

    def _insert_recursive(self, root, node):
        if node.key < root.key:
            if root.left is None:
                root.left = node
                node.parent = root
            else:
                self._insert_recursive(root.left, node)
        else:
            if root.right is None:
                root.right = node
                node.parent = root
            else:
                self._insert_recursive(root.right, node)

    def _fix_insert(self, node):
        while node != self.root and node.parent.color == 'red':
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right
                if uncle is not None and uncle.color == 'red':
                    node.parent.color = 'black'
                    uncle.color = 'black'
                    node.parent.parent.color = 'red'
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        node = node.parent
                        self._left_rotate(node)
                    node.parent.color = 'black'
                    node.parent.parent.color = 'red'
                    self._right_rotate(node.parent.parent)
            else:
                uncle = node.parent.parent.left
                if uncle is not None and uncle.color == 'red':
                    node.parent.color = 'black'
                    uncle.color = 'black'
                    node.parent.parent.color = 'red'
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        node = node.parent
                        self._right_rotate(node)
                    node.parent.color = 'black'
                    node.parent.parent.color = 'red'
                    self._left_rotate(node.parent.parent)
        self.root.color = 'black'

    def _left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left is not None:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _right_rotate(self, x):
        y = x.left
        x.left = y.right
        if y.right is not None:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    def search(self, key):
        return self._search_recursive(self.root, key)

    def _search_recursive(self, node, key):
        if node is None or node.key == key:
            return node
        if key < node.key:
            return self._search_recursive(node.left, key)
        else:
            return self._search_recursive(node.right, key)

    def remove(self, key):
        node = self.search(key)
        if node is None:
            return
        self._remove_node(node)

    def _remove_node(self, node):
        if node.left is None or node.right is None:
            child = node.right if node.left is None else node.left
            if node.parent is None:
                self.root = child
            elif node == node.parent.left:
                node.parent.left = child
            else:
                node.parent.right = child
            if child is not None:
                child.parent = node.parent
            if node.color == 'black':
                self._fix_delete(child, node.parent)
        else:
            successor = self._find_successor(node)
            node.key = successor.key
            self._remove_node(successor)

    def _fix_delete(self, node, parent):
        while node != self.root and (node is None or node.color == 'black'):
            if node == parent.left:
                sibling = parent.right
                if sibling.color == 'red':
                    sibling.color = 'black'
                    parent.color = 'red'
                    self._left_rotate(parent)
                    sibling = parent.right
                if (sibling.left is None or sibling.left.color == 'black') and \
                        (sibling.right is None or sibling.right.color == 'black'):
                    sibling.color = 'red'
                    node = parent
                    parent = node.parent
                else:
                    if sibling.right is None or sibling.right.color == 'black':
                        sibling.left.color = 'black'
                        sibling.color = 'red'
                        self._right_rotate(sibling)
                        sibling = parent.right
                    sibling.color = parent.color
                    parent.color = 'black'
                    sibling.right.color = 'black'
                    self._left_rotate(parent)
                    node = self.root
            else:
                sibling = parent.left
                if sibling.color == 'red':
                    sibling.color = 'black'
                    parent.color = 'red'
                    self._right_rotate(parent)
                    sibling = parent.left
                if (sibling.right is None or sibling.right.color == 'black') and \
                        (sibling.left is None or sibling.left.color == 'black'):
                    sibling.color = 'red'
                    node = parent
                    parent = node.parent
                else:
                    if sibling.left is None or sibling.left.color == 'black':
                        sibling.right.color = 'black'
                        sibling.color = 'red'
                        self._left_rotate(sibling)
                        sibling = parent.left
                    sibling.color = parent.color
                    parent.color = 'black'
                    sibling.left.color = 'black'
                    self._right_rotate(parent)
                    node = self.root
        if node is not None:
            node.color = 'black'

    def _find_successor(self, node):
        if node.right is not None:
            return self._find_min(node.right)
        parent = node.parent
        while parent is not None and node == parent.right:
            node = parent
            parent = parent.parent
        return parent

    def _find_min(self, node):
        while node.left is not None:
            node = node.left
        return node

    def pretty_print(self):
        def _print_tree(node, prefix="", is_left=True):
            if node is not None:
                print(prefix + ("|-- " if is_left else "`-- ") + str(node.key) + (
                    " (R)" if node.color == 'red' else " (B)"))
                _print_tree(node.left, prefix + ("|   " if is_left else "    "), True)
                _print_tree(node.right, prefix + ("|   " if is_left else "    "), False)

        _print_tree(self.root)


if __name__ == "__main__":
    tree = RedBlackTree()

    tree.insert(55)
    tree.insert(40)
    tree.insert(65)

    print(tree.search(40).key)

    tree.pretty_print()
