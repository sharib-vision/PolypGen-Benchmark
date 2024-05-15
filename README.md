# PolypGen-Benchmark

### Brief description:

<p align="justify">This is a benchmark code used for polyp segmentation reported in <a href="https://arxiv.org/pdf/2106.04463.pdf">PolypGen: A multi-center polyp detection andsegmentation dataset for generalisabilityassessment</a>.
    
**Cite:**

[1] Ali, S. et al. A multi-centre polyp detection and segmentation dataset for generalisability assessment. Sci Data 10, 75 (2023). https://doi.org/10.1038/s41597-023-01981-y

[2] Ali, S., Ghatwary, N., Jha, D. et al. Assessing generalisability of deep learning-based polyp detection and segmentation methods through a computer vision challenge. Sci Rep 14, 2032 (2024). https://doi.org/10.1038/s41598-024-52063-x

``Bibtex citations``
```
[1] @ARTICLE{Ali2023-qo,
  title    = "A multi-centre polyp detection and segmentation dataset for
              generalisability assessment",
  author   = "Ali, Sharib and Jha, Debesh and Ghatwary, Noha and Realdon,
              Stefano and Cannizzaro, Renato and Salem, Osama E and Lamarque,
              Dominique and Daul, Christian and Riegler, Michael A and Anonsen,
              Kim V and Petlund, Andreas and Halvorsen, P{\aa}l and Rittscher,
              Jens and de Lange, Thomas and East, James E",
  journal  = "Sci. Data",
  volume   =  10,
  number   =  1,
  pages    = "75",
  month    =  feb,
  year     =  2023,
  language = "en"
}

[2] @article{Ali2022AssessingGO,
  title={Assessing generalisability of deep learning-based polyp detection and segmentation methods through a computer vision challenge},
  author={Sharib Ali and Noha M. Ghatwary and Debesh Jha and Ece Isik-Polat and Gorkem Polat and Chen Yang and Wuyang Li and Adrian Galdran and Miguel {\'A}ngel Gonz{\'a}lez Ballester and Vajira Lasantha Thambawita and Steven Hicks and Sahadev Poudel and Sang-Woong Lee and Ziyi Jin and Tianyuan Gan and Chen-Ping Yu and Jiangpeng Yan and Doyeob Yeo and Hyunseok Lee and Nikhil Kumar Tomar and Mahmood Haithmi and Amr Ahmed and M. Riegler and Christian Daul and Paal Halvorsen and Jens Rittscher and Osama E. Salem and Dominique Lamarque and Renato Cannizzaro and Stefano Realdon and Thomas de Lange and James Edward East},
  journal={Scientific Reports},
  year={2022},
  volume={14},
  url={https://api.semanticscholar.org/CorpusID:247084187}
}
```



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
