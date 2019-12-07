This is matrix multiplication app for showing CUDA acceleration against CPU.

Here is some results of running this application on Intel Xeon 1650v3 and NVIDIA GTX 1060.

Results from 1000 size to 5

| Size | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 5000 | 5500 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CPU (ms) | 1736 | 1611 | 1550 | 1556 | 1708 | 1923 | 1726 | 1843 | 1789 | 1744 |
| GPU (ms) | 1432 | 1447 | 1468 | 1437 | 1683 | 1893 | 1686 | 1774 | 1678 | 1689 |
| Acceleration | 1432 | 1447 | 1468 | 1437 | 1683 | 1893 | 1686 | 1774 | 1678 | 1689 |
