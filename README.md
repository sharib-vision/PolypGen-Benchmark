# PolypGen-Benchmark

### Brief description:

<p align="justify">This is a benchmark code used for polyp segmentation reported in <a href="https://arxiv.org/pdf/2106.04463.pdf">PolypGen: A multi-center polyp detection andsegmentation dataset for generalisabilityassessment</a>.
    
**Cite:**
Ali, S. et al. PolypGen: A multi-center polyp detection and segmentation dataset for generalisability assessment. arXiv 2021. Link: https://arxiv.org/abs/2106.04463



## Installation
Requirements:

- Python 3.6.5+
- Pytorch (version 1.4.0)
- Torchvision (version 0.5.0)
- Visdom and many (will update it!!!)

1. Clone this repository
    
    `git clone https://github.com/sharibox/PolypGen-Benchmark.git`
    
    `cd PolypGen-Benchmark`

2. Goto your environment with installations

3. Clone additional UNet backbone repository
   
    `git clone https://github.com/mkisantal/backboned-unet.git`
    
    `cd backboned-unet`
    
    `pip install .`

**Metrics**
- Please use the metrics provided [here](https://github.com/sharibox/PolypGen-Benchmark/blob/main/metrics/compute_seg.py)
    
## Statement & Disclaimer
This project is for research purpose only and may have included third party educational/research codes. The authors do not take any liability regarding any commercial usage. For any other questions please contact [ali.sharib2002@gmail.com](mailto:ali.sharib2002@gmail.com).
