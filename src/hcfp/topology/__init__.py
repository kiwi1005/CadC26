"""Topology learning and hard sequence-pair packing primitives."""

from hcfp.topology.longest_path_pack import (
    longest_path_coordinates,
    longest_path_pack,
    pack_sequence_pair,
)
from hcfp.topology.permutation_head import (
    DualPermutationHead,
    greedy_hard_assignment,
    hard_permutation,
    sinkhorn,
)
from hcfp.topology.preplaced_adaptation import (
    PreplacedCompatibility,
    PreplacedConflict,
    adapt_preplaced_topology,
    check_preplaced_compatibility,
    copy_preplaced_targets,
)
from hcfp.topology.relation_labels import (
    INVERSE_RELATION,
    REL_ABOVE,
    REL_BELOW,
    REL_DOWN,
    REL_LEFT,
    REL_RIGHT,
    REL_UP,
    antisymmetry_loss,
    partial_label_nll,
    relation_mask_from_rectangles,
)
from hcfp.topology.sequence_pair import (
    REL_NONE,
    SequencePairTopology,
    decode_sequence_pair,
)


__all__ = [
    "INVERSE_RELATION",
    "REL_ABOVE",
    "REL_BELOW",
    "REL_DOWN",
    "REL_LEFT",
    "REL_NONE",
    "REL_RIGHT",
    "REL_UP",
    "DualPermutationHead",
    "PreplacedCompatibility",
    "PreplacedConflict",
    "SequencePairTopology",
    "adapt_preplaced_topology",
    "antisymmetry_loss",
    "check_preplaced_compatibility",
    "copy_preplaced_targets",
    "decode_sequence_pair",
    "greedy_hard_assignment",
    "hard_permutation",
    "longest_path_coordinates",
    "longest_path_pack",
    "pack_sequence_pair",
    "partial_label_nll",
    "relation_mask_from_rectangles",
    "sinkhorn",
]
