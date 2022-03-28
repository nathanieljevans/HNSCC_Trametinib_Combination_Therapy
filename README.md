# Patient Specific Potential for Trametinib Therapy in HNSCC 

`Combination Index` analysis

Data can be downloaded from [onedrive](https://ohsuitg-my.sharepoint.com/:u:/r/personal/evansna_ohsu_edu/Documents/00_EVANS/MYLES_TRAMETINIB_DATA_REPO/bliss_beta_all.zip?csf=1&web=1&e=37ecVB). Email `evansna@ohsu.edu` for permissions or questions. 

Use `environment.yml` to create a conda environment with necesary dependencies. 

To run the combination index results: 

```bash 
$ python Calculate_CI_values.py --ICxx 0.6 --out ./../output/results.csv
```

An overview of the methods are available in `tutorial.ipynb`. 