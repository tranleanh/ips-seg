# IPS-Seg

[![Preprint](https://img.shields.io/badge/Preprint-arXiv-red)](https://arxiv.org/html/2607.03754v1)

Exploring SAM Supervision for Fine-Grained UAV Target Segmentation under Data Scarcity

Author: Le-Anh Tran

## Framework

(to be updated)

## Inference: Two-stage SAM3
- Download SAM3 weight file from [facebook/sam3](https://huggingface.co/facebook/sam3) and save it in "models".
- Run two-stage SAM3 (SAM3_2S) on input folder "imgs/inputs" and save results to "imgs/inputs_pred":
```bashrc
python inference_SAM3_2S.py --in_dir imgs/inputs --out_dir imgs/inputs_pred
```
- See more options in [inference_SAM3_2S.py](https://github.com/tranleanh/ips-seg/blob/main/inference_SAM3_2S.py)

## Inference: Two-stage IPS-Seg
- Run two-stage IPS-Seg (IPS-Seg_2S):
```bashrc
(to be updated)
```

## Results

<p align="center">
<img src="docs/mask_refinement_results.png" width="1000">
</p>

## Citation
```bibtex
(to be updated)
```

Have fun!

LA Tran
