class Node:
    def __init__(self, key):
        self.key = key
        self.prev = None
        self.next = None


class DoubleLinkedList:
    def __init__(self):
        self.head = Node(None)
        self.tail = Node(None)
        self.head.next = self.tail
        self.tail.prev = self.head

    def insert(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def delete(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def get_delete(self, key):
        node = self.get(key)
        if node:
            self.delete(node)
        return node

    def __iter__(self):
        node = self.head.next
        while node.next is not None:
            yield node
            node = node.next

    def get(self, key, default=None):
        for node in self:
            if node.key == key:
                return node
        return default

    def __str__(self):
        s = []
        for node in self:
            s.append(str(node.key))

        return " -> ".join(s)


if __name__ == "__main__":
    dll = DoubleLinkedList()
    dll.insert(Node(1))
    dll.insert(Node(2))
    dll.insert(Node(3))

    print(dll)

    dll.get_delete(2)
    print(dll)
