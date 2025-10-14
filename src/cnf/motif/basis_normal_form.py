from .atomic_motif import FractionalMotif, DiscretizedMotif

class BasisNormalForm():
    """Implements methods for taking a list of atomic positions
    in fractional coordinates and producing the Basis Normal Form string
    as described in the section "Representation of Crystalline Atomic Bases" on
    pp. 52 of David Mrdjenovich's thesis.
    """

    def __init__(self, coord_list, element_list, delta):
        self.coord_list = coord_list
        self.elements = [str(e) for e in element_list]
        self.delta = delta

    def to_motif(self):
        frac_coords = [c / self.delta for c in self.coord_list]
        separated_coord_lists = [frac_coords[start_idx:start_idx+3] for start_idx in range(0, len(frac_coords), 3)]
        separated_coord_lists = [[0, 0, 0]] + separated_coord_lists

        return FractionalMotif.from_elements_and_positions(self.elements, separated_coord_lists)
    
    def to_discretized_motif(self):
        separated_coord_lists = [self.coord_list[start_idx:start_idx+3] for start_idx in range(0, len(self.coord_list), 3)]
        separated_coord_lists = [[0, 0, 0]] + separated_coord_lists

        return DiscretizedMotif.from_elements_and_positions(self.elements, separated_coord_lists, self.delta)

    def to_dict(self):
        return {
            "delta": self.delta,
            "elements": [str(e) for e in self.elements],
            "coords": self.coord_list
        }
    
    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            d["coords"],
            d["elements"],
            d["delta"]
        )

    def __repr__(self):
        return f"BasisNormalForm({self.coord_list},elements={self.elements},delta={self.delta})"
