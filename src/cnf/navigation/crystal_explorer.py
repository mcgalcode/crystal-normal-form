import json
import time
from ..unit_cell import UnitCell
from .crystal_map import CrystalMap
from .neighbor_finder import NeighborFinder
from ..crystal_normal_form import CrystalNormalForm
from sortedcontainers import SortedSet
from .search_filters import SearchFilter
from .search_objectives import SearchObjective
from .score_functions import ScoreFunction


class CrystalExplorer():

    def __init__(self, cmap: CrystalMap, search_filter: SearchFilter, score_fn: ScoreFunction, skip_scoring=False, preload_scores: dict = None):
        self.map = cmap
        self.score_function = score_fn
        self.search_filter = search_filter

        # Track all points (scored or not) and their exploration state
        self._unexplored_pts = set()
        self._explored_pts = set()

        # Track scores (only for points that have been scored)
        if preload_scores is None:
            self.scores = {}
        else:
            self.scores = preload_scores

        self.unexplored_score_list = SortedSet()
        self.explored_score_list = SortedSet()

        if not skip_scoring:
            for pt in cmap.all_node_ids():
                self._unexplored_pts.add(pt)
                pt = cmap.get_point_by_id(pt)
                self.score_pt(pt)
    
    def explore_point(self, point_id: int):
        pt = self.map.get_point_by_id(point_id)
        nf = NeighborFinder(pt)
        nb_pts = nf.find_neighbors()
        new_ids = []
        for nb_pt in nb_pts:
            if self.search_filter.should_add_pt(nb_pt):
                if nb_pt not in self.map:
                        nid = self.map.add_point(nb_pt)
                        new_ids.append(nid)
                        self.score_pt(nb_pt, explored=False)
                self.map.add_connection(pt, nb_pt)                    
            else:
                pass
                # print("Skipping point outside valid bounds...")

        self._set_pt_explored(point_id)
        return new_ids
    
    def search(self, search_object: SearchObjective):

        tries = 0
        prev_len = len(self.map)
        start_time = time.perf_counter()
        while not search_object.objective_complete(self):
            if tries % 10 == 0:
                end_time = time.perf_counter()
                print("===========================================================")
                print(f"Starting round {tries} of searching for endpts, map has {len(self.map)} pts")
                print(f"Current best score: {self.best_current_score()}")
                curr_len = len(self.map)
                diff = curr_len - prev_len
                print(f"Added {diff} points in last 10 rounds")
                elapsed_time = end_time - start_time
                print(f"10 tries took {elapsed_time:.6f} seconds")
                prev_len = curr_len
                start_time = time.perf_counter()

            total_added = 0
            
            for pt_id in self.unexplored_points():
                # pt = self.map.get_point_by_id(pt_id)
                # print(f"Exploring pt: {pt.coords} (score: {self.score_for_point(pt_id)})")
                new_ids = self.explore_point(pt_id)
                diff = len(new_ids)
                total_added += diff
                # print(f"Added {diff} pts!")
                if diff > 0:
                    break
            tries += 1
            if total_added == 0:
                print(f"Exhausted map boundaries!")
    
    def score_pt(self, pt: CrystalNormalForm, explored=False):
        score = self.score_function.score(pt)
        pt_id = self.map.get_point_id(pt)
        self._add_scored_pt(pt_id, score, explored)
        return score
    
    def _add_scored_pt(self, pt_id, score, explored):
        if explored:
            self._set_pt_explored(pt_id)
        else:
            self._set_pt_unexplored(pt_id)
        self.set_pt_score(pt_id, score)
    
    def set_pt_score(self, pt_id, score):
        if self.is_id_explored(pt_id):
            if pt_id in self.scores:
                self.explored_score_list.remove(self._get_score_item(pt_id))
            self.scores[pt_id] = score
            self.explored_score_list.add(self._get_score_item(pt_id))
        else:
            if pt_id in self.scores:
                self.unexplored_score_list.remove(self._get_score_item(pt_id))
            self.scores[pt_id] = score
            self.unexplored_score_list.add(self._get_score_item(pt_id))
        
    def _get_score_item(self, pt_id):
        return (self.scores[pt_id], pt_id)

    def _set_pt_explored(self, pt_id):
        if pt_id in self._unexplored_pts:
            self._unexplored_pts.remove(pt_id)
            if pt_id in self.scores:
                self.unexplored_score_list.remove(self._get_score_item(pt_id))

        self._explored_pts.add(pt_id)
        if pt_id in self.scores:
            self.explored_score_list.add(self._get_score_item(pt_id))
    
    def _set_pt_unexplored(self, pt_id):
        if pt_id in self._explored_pts:
            self._explored_pts.remove(pt_id)
            if pt_id in self.scores:
                self.explored_score_list.remove(self._get_score_item(pt_id))

        self._unexplored_pts.add(pt_id)
        if pt_id in self.scores:
            self.unexplored_score_list.add(self._get_score_item(pt_id))
    
    def unexplored_points(self):
        return [i[1] for i in self.unexplored_score_list]

    def best_current_score(self):
        all_scores = self.unexplored_score_list.union(self.explored_score_list)
        if len(all_scores) == 0:
            return None
        else:
            return all_scores[0][0]
    
    def score_for_point(self, pt_id: int):
        return self.scores[pt_id]
    
    def is_point_explored(self, point: CrystalNormalForm):
        pid = self.map.get_point_id(point)
        return self.is_id_explored(pid)
    
    def is_id_explored(self, id: int):
        if id not in self.map.all_node_ids():
            raise ValueError(f"Tried to check if nonexistant node id {id} was explored")
        return id in self._explored_pts
    
    @classmethod
    def from_dict(cls, d):
        cmap = CrystalMap.from_dict(d["crystal_map"])
        scores = { int(nid): score for nid, score in d["scores"].items() }
        e = cls(
            cmap,
            None,
            None,
            skip_scoring=True,
            preload_scores=scores
        )
        for nid in cmap.all_node_ids():
            e._set_pt_unexplored(nid)
        print("Finished setting unexplored")
        for nid in d["explored_ids"]:
            # print(f"{nid} in is_explored")
            e._set_pt_explored(nid)
        print("Finished setting explored")

        return e
    
    @classmethod
    def from_json(cls, fname):
        with open(fname, 'r+') as f:
            d = json.load(f)
            return cls.from_dict(d)


    def to_dict(self):
        return {
            "crystal_map": self.map.as_dict(),
            "scores": self.scores,
            "explored_ids": list(self._explored_pts),
        }
        
    def to_json(self, fname: str):
        d = self.to_dict()
        with open(fname, 'w+') as f:
            json.dump(d, f)
    