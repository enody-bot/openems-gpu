"""Minimal S-expression reader for KiCad files. Atoms stay str; quoted strings are QStr; lists are lists."""
import re


class QStr(str):
    """A string atom that was quoted in the file."""


_TOKEN = re.compile(r'\s*(?:(\()|(\))|"((?:[^"\\]|\\.)*)"|([^\s()"]+))', re.S)


def parse(text):
    stack = [[]]
    pos, n = 0, len(text)
    while pos < n:
        m = _TOKEN.match(text, pos)
        if not m:
            if text[pos:].strip() == '':
                break
            raise ValueError('bad s-expression near %r' % text[pos:pos + 40])
        pos = m.end()
        if m.group(1):
            stack.append([])
        elif m.group(2):
            lst = stack.pop()
            stack[-1].append(lst)
        elif m.group(3) is not None:
            stack[-1].append(QStr(m.group(3).replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')))
        else:
            stack[-1].append(m.group(4))
    if len(stack) != 1:
        raise ValueError('unbalanced s-expression')
    return stack[0]


def kv(node, key):
    """The first child list of `node` whose head is `key`, or None."""
    for c in node[1:]:
        if isinstance(c, list) and c and c[0] == key:
            return c
    return None


def kvs(node, key):
    return [c for c in node[1:] if isinstance(c, list) and c and c[0] == key]
