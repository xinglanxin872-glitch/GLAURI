# Model variants

GLAURI is the main model. See [training targets](../colabel/README.md#training-targets)
for annotation preparation.

| Config ID | Display name | Changes from GLAURI |
| --- | --- | --- |
| [GLAURI](configs/GLAURI.json) | GLAURI | Main model. |
| [w_o_structure_set_weighting](configs/w_o_structure_set_weighting.json) | w/o structure-set weighting | Sample without structure-set weighting. |
| [w_o_positive_ratio_weighting](configs/w_o_positive_ratio_weighting.json) | w/o positive ratio weighting | Use uniform weights across active tasks. |
| [w_o_PLM](configs/w_o_PLM.json) | w/o PLM | Remove ESM input embeddings. |
| [w_o_struct](configs/w_o_struct.json) | w/o struct | Remove structural features, including RSA; retain sequence adjacency and disable coordinate noise. |
| [w_o_co_label](configs/w_o_co_label.json) | w/o co-label | Use native contact annotations only. |
| [Protein](configs/Protein.json) | Protein | Optimize only the protein-interface task. |
| [Nucleic_acid](configs/Nucleic_acid.json) | Nucleic acid | Optimize only the nucleic-acid-interface task. |
| [Ion](configs/Ion.json) | Ion | Optimize only the ion-interface task. |
| [Ligand](configs/Ligand.json) | Ligand | Optimize only the ligand-interface task. |

w/o multi-task combines four matching single-task heads, not another checkpoint.
