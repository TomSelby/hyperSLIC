# HyperSLIC

HyperSLIC is an adaptation of the simple linear iterative clustering (SLIC) algorithm widely used on remote sensing images for the clustering of high-dimensional microscopy datasets.
## Installation and Dependancies
### Python Dependancies
hyperSLIC is largely built upon [pyxem](https://pyxem.readthedocs.io/en/stable/index.html) and [hyperspy](https://hyperspy.org/hyperspy-doc/current/index.html),
pip installing pyxem should provide both of these:
```
pip install pyxem
```
## Usage
```python
import hyperSLIC
```
Separate the 'wheat' from the 'chaff' given a certain threshold of the dynamic range or variance.
```python
wheat = wheat_from_chaff(raveled,98,'range')
```
Make a hyperSLIC object
```python
test = hyperSLIC.SLIC(wheat_hs,method,cluster_number,m_value,searchspace)
```
Run the clustering in a loop
```python
test.find_closest_centeroid()
test.update_centeroids()
```
## Demo
Please follow the ```demo.ipynb``` jupyter notebook for detailed instructions on use, including expected runtime and outputs. An example scanning electron diffraction (SED) dataset found in [/data](https://pages.github.com/)
## Contributing

Feel free to raise an issue if you experience issues or wish to see new functionality implemented.

## References
1. Y. Shi, W. Wang, Q. Gong and D. Li, The Journal of Engineering, 2019, 2019, 6675–6679. (https://doi.org/10.1049/joe.2019.0240)
2. R. Achanta, A. Shaji, K. Smith, A. Lucchi, P. Fua and S. Süsstrunk, IEEE Trans Pattern Anal Mach Intell, 2012, 34, 2274–2282. (https://doi.org/10.1109/TPAMI.2012.120)
