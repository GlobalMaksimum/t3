# t3

### ! Since this project contains submodule make sure you clone the project with !

```
git clone --recursive https://github.com/GlobalMaksimum/t3
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
[Click for document](https://github.com/GlobalMaksimum/t3/blob/master/3T%20Rapor%20Taslag%CC%86%C4%B1.pdf)

### Homepage
[Click for website](http://turkiyeteknolojitakimi.org/)

### Initial Report Template
[Click for document](https://www.teknofestistanbul.org/Content/files/2019_satnameler/yapay_zeka_otr_sablon_14.docx)

## Evaluation
*_txt file format:_* 

```
image_name,x1,y1,x2,y2,class_id,...
...
```

`python evaluate.py gt.txt preds.txt --threshold 0.6`
