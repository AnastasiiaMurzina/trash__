class Data(object):
    def __init__(self, name):
        self.__name = name
        self.__links = set()

    @property
    def name(self):
        return self.__name

    @property
    def links(self):
        return set(self.__links)

    def add_link(self, other):
        self.__links.add(other)
        other.__links.add(self)

def connected_components(nodes):
    result = []
    nodes = set(nodes)
    while nodes:
        n = nodes.pop()
        group = {n}
        queue = [n]
        while queue:
            n = queue.pop(0)
            neighbors = n.links
            neighbors.difference_update(group)
            nodes.difference_update(neighbors)
            group.update(neighbors)
            queue.extend(neighbors)
        result.append(group)
    return result


if __name__ == "__main__":
    with open('input.txt', 'r') as f1:
        n, m = f1.readline().split()
        n, m = int(n), int(m)
        nodes = []
        for i in range(n):
            exec(str('a'+str(i+1)+'=Data("a'+str(i+1)+'")'))
            eval('nodes.append(a'+str(i+1)+')')

        for i in range(m):
            bond = f1.readline().split()
            eval('a'+bond[0]+'.add_link(a'+bond[1]+')')

    number = 1
    comp = connected_components(nodes)
    with open('output.txt', 'w') as f2:
        if len(comp)!=2:
            f2.write('-1')
        else:
            for components in comp:
                # names = sorted(node.name for node in components)
                names = [node.name for node in components]
                names = [i[1::] for i in names]
                names = " ".join(names)
                if number==1:
                    f2.write(str(len(names))+'\n')
                f2.write(' '.join(str(names))+'\n')
                number += 1