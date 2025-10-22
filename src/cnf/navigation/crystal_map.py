from ..crystal_normal_form import CrystalNormalForm

import rustworkx as rwx

class CrystalMap():

    @classmethod
    def from_cnfs(cls, cnfs: list[CrystalNormalForm]):
        cnf = cnfs[0]
        crys_map = cls(cnf.xi, cnf.delta, cnf.elements)
        for cnf in cnfs:
            crys_map.add_point(cnf)
        return crys_map
    
    @classmethod
    def from_dict(cls, d: dict):
        cmap = cls(d["xi"], d["delta"], d["elements"])
        node_ids_and_pts = [(int(nid), cnf) for nid, cnf in d["graph"]["nodes"].items()]
        node_ids_and_pts = sorted(node_ids_and_pts)
        for (nid, cnf) in node_ids_and_pts:
            cmap.add_point(CrystalNormalForm.from_tuple(cnf, d["elements"], d["xi"], d["delta"]))
        
        for e in d["graph"]["edges"]:
            cmap.add_connection_by_ids(e[0], e[1])
        return cmap

    def __init__(self, xi: float, delta: int, element_list: list[str]):
        self.xi = xi
        self.delta = delta
        self.element_list = element_list

        self._graph = rwx.PyGraph()
        self._all_points_set = set()
        self._id_lookup = {}

    def validate_point(self, cnf: CrystalNormalForm):
        if not isinstance(cnf, CrystalNormalForm):
            raise ValueError(f"Can't use point of type {type(cnf)} with CrystalMap - must use CrystalNormalForm instance.")

        if cnf.xi != self.xi:
            raise ValueError(f"Tried to add CNF with incompatible xi to CrystalMap (incoming: {cnf.xi}, map: {self.xi})")
        
        if cnf.delta != self.delta:
            raise ValueError(f"Tried to add CNF with incompatible delta to CrystalMap (incoming: {cnf.delta}, map: {self.delta})")

        if tuple(cnf.elements) != tuple(self.element_list):
            raise ValueError(f"Tried to add CNF with incompatible element list to CrystalMap (incoming: {cnf.elements}, map: {self.element_list})")

    def add_point(self, point: CrystalNormalForm):
        # self.validate_point(point)
        if point in self._all_points_set:
            return None
        node_id = self._graph.add_node(point)
        self._all_points_set.add(point)
        self._id_lookup[point] = node_id
        return node_id
    
    def all_points(self):
        return self._graph.nodes()
    
    def all_node_ids(self):
        return list(self._graph.node_indices())
    
    def get_point_id(self, point: CrystalNormalForm):
        return self._id_lookup.get(point)
    
    def get_point_ids(self, *points: list[CrystalNormalForm]):
        return tuple([self.get_point_id(p) for p in points])

    def get_point_by_id(self, id: int):
        return self._graph[id]
    
    def remove_point(self, point: CrystalNormalForm):
        if point not in self:
            return None
        point_id = self.get_point_id(point)
        self._graph.remove_node(point_id)
        self._all_points_set.remove(point)
        del self._id_lookup[point]
        return point_id

    def add_connection(self, pt1: CrystalNormalForm, pt2: CrystalNormalForm):
        id1, id2 = self.get_point_ids(pt1, pt2)
        return self.add_connection_by_ids(id1, id2)

    def add_connection_by_ids(self, id1, id2):
        if self._graph.has_edge(id1, id2):
            return False
        self._graph.add_edge(id1, id2, 1)
        return True
        
    def remove_connection(self, pt1: CrystalNormalForm, pt2: CrystalNormalForm):
        if not self.connection_exists(pt1, pt2):
            return False
        id1, id2 = self.get_point_ids(pt1, pt2)
        self._graph.remove_edge(id1, id2)
        return True

    def connection_exists(self, pt1: CrystalNormalForm, pt2: CrystalNormalForm):
        if pt1 not in self or pt2 not in self:
            raise ValueError(f"Tried to look for connections involving a nonexistent point!")
        
        id1, id2 = self.get_point_ids(pt1, pt2)
        return self.connection_exists_by_id(id1, id2)
    
    def connection_exists_by_id(self, id1: int, id2: int):
        return self._graph.has_edge(id1, id2)
    
    def __contains__(self, item):
        if not isinstance(item, CrystalNormalForm):
            raise ValueError(f"Can't check if item {type(item)} is in CrystalMap - must use CrystalNormalForm instance.")
        
        return item in self._all_points_set
    
    def __len__(self):
        return len(self._all_points_set)

    def as_dict(self):
        # Convert to a dictionary representation
        graph_dict = {
            "nodes": {},
            "edges": []
        }
        for node_index in self._graph.node_indices():
            # Get the node's data payload
            cnf_pt: CrystalNormalForm = self._graph.get_node_data(node_index)
            graph_dict["nodes"][node_index] = cnf_pt.coords
        
        for edge in self._graph.edge_list():
            graph_dict["edges"].append(edge)

        return {
            "xi": self.xi,
            "delta": self.delta,
            "elements": [str(e) for e in self.element_list],
            "graph":  graph_dict
        }