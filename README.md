# HyperSLIC

HyperSLIC is an adaptation of the simple linear iterative clustering (SLIC) algorithm widely used on remote sensing images for the clustering of high-dimensional microscopy datasets.

## Usage
The ```SLIC.ipynb``` notebook gives an example of how the package is applied to a scanning electron diffraction (SED) dataset.
```python
import hyperSLIC
```
Separate the 'wheat' from the 'chaff'
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

## Contributing

We're happy for users to use the code on their own data however please keep this repo and results confidential until publication of the algorithm (should be soon).

Feel free to raise an issue if you experience issues or wish to see new functionality implemented.

## References
1. Y. Shi, W. Wang, Q. Gong and D. Li, The Journal of Engineering, 2019, 2019, 6675–6679. (https://doi.org/10.1049/joe.2019.0240)
2. R. Achanta, A. Shaji, K. Smith, A. Lucchi, P. Fua and S. Süsstrunk, IEEE Trans Pattern Anal Mach Intell, 2012, 34, 2274–2282. (https://doi.org/10.1109/TPAMI.2012.120)
