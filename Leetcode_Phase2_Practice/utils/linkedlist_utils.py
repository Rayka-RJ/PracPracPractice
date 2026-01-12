from typing import List, Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def build_linked_list(nums: List[int]) -> Optional[ListNode]:
    """从列表构建链表"""
    dummy = ListNode()
    current = dummy
    for num in nums:
        current.next = ListNode(num)
        current = current.next
    return dummy.next

def linked_list_to_list(head: Optional[ListNode]) -> List[int]:
    """将链表转换为列表"""
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result

def print_linked_list(head: Optional[ListNode]) -> str:
    """打印链表"""
    vals = []
    while head:
        vals.append(str(head.val))
        head = head.next
    return ' -> '.join(vals) if vals else 'Empty'

# 带random指针的节点
class RandomNode:
    def __init__(self, x: int, next: 'RandomNode' = None, random: 'RandomNode' = None):
        self.val = int(x)
        self.next = next
        self.random = random

def build_random_list(data: List[tuple]) -> Optional[RandomNode]:
    """
    构建带random指针的链表
    data: List of tuples like [(val, random_index), ...]
    Example: [(7, None), (13, 0), (11, 4), (10, 2), (1, 0)]
    """
    if not data:
        return None
    nodes = [RandomNode(val) for val, _ in data]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    for i, (_, rand_i) in enumerate(data):
        if rand_i is not None:
            nodes[i].random = nodes[rand_i]
    return nodes[0]
