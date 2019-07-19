# t3

## Installation

### ! Since this project contains submodule make sure you clone the project with !

```
git clone --recursive https://github.com/GlobalMaksimum/t3
```

#### Requirements (Docker version will be released)

Linux
Python 3.5+
PyTorch 1.0+ or PyTorch-nightly
GCC 4.9+
mmcv

- Install Cython if not already exists.

```
conda install cython
```

- Install PyTorch stable or nightly and torchvision following the official instructions.

```
cd mmdetection
```
- Compile cuda extensions.

```
./compile.sh
```

- Install mmdetection (other dependencies will be installed automatically).

```
python setup.py develop
```


## Project Structure

```
.
├── data
│   ├── external
│   └── t3-data
│       └── gonderilecek_veriler
│           ├── B160519_V1_K1
│           ├── T190619_V1_K1
│           ├── T190619_V2_K1
│           └── T190619_V3_K1
├── models
├── notebooks
├── reports
├── src
    ├── dataloaders
    └── utils
```

## References

### Specification
[Click for PDF](https://www.teknofestistanbul.org/Content/files/2019_satnameler/Yapay_Zeka_Yarismasi_Sartname_05.pdf)

### Report Document
[Click for document](https://docs.google.com/document/d/1vh7A7FA0Czu0vA2m6x8PM3eM-67qG2w1d6ymAlVYZCY/)

### Homepage
[Click for website](http://turkiyeteknolojitakimi.org/)

### Initial Report Template
[Click for document](https://www.teknofestistanbul.org/Content/files/2019_satnameler/yapay_zeka_otr_sablon_14.docx)
