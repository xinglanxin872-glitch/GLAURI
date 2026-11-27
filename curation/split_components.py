"""Connected components shared by sequence and native co-label groups."""
import pickle
from pathlib import Path


def chain_key(key):
    fields = str(key).split('/')
    return (fields[1] + '_' + fields[2].split(':')[0]).lower()


class Components:
    def __init__(self):
        self.parent = {}

    def find(self, value):
        value = str(value).lower()
        self.parent.setdefault(value, value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def load_components(coarse_path, native_path):
    components = Components()
    memberships = {}
    centers = {}
    for cid, line in enumerate(Path(coarse_path).read_text().splitlines()):
        members = line.lower().split()
        if not members:
            continue
        centers[cid] = members[0]
        for member in members:
            memberships.setdefault(member, set()).add(cid)
            components.union(members[0], member)
    with Path(native_path).open('rb') as stream:
        native = pickle.load(stream)
    anchors = {}
    for key, value in native.items():
        chain, cluster = chain_key(key), str(value['cluster'])
        if cluster in anchors:
            components.union(chain, anchors[cluster])
        else:
            anchors[cluster] = chain
    return components, native, memberships, centers
