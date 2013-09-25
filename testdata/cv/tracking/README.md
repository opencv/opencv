## Testdata for tracking module

All image files and ground truth are taken from.

The original images sequences are combined into video files with this command:
ffmpeg  -i %4d.jpg -vcodec libvpx -crf 4 -b 1M testset.webm

```
@inproceedings{ WuLimYang13,
  Title = {Online Object Tracking: A Benchmark},
  Author = {Yi Wu and Jongwoo Lim and Ming-Hsuan Yang},
  Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  Year = {2013}
}
```
WEB SITE: https://sites.google.com/site/trackerbenchmark/benchmarks
