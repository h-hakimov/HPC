This is matrix multiplication app for showing CUDA acceleration against CPU.

Here is some results of running this application on Intel Xeon 1650v3 and NVIDIA GTX 1060.

Results from 1000 size to 3000

| Size | 1000 | 1500 | 2000 | 2500 | 3000 |
| --- | --- | --- | --- | --- | --- |
| CPU (ms) | 925 | 3565 | 8996 | 19979 | 37276 |
| GPU (ms) | 5 | 21 | 41 | 80 | 136 |
| Acceleration | 472 | 167 | 214 | 248 | 273 |
