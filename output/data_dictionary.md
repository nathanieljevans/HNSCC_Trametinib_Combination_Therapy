# Results Data Dictionary 

The results of running the script:   

```bash 

inhibitory_constant=0.5
save_path=./../output/results.csv

$ python Calculate_CI_values.py --ICxx $inhibitory_constant --out $save_path

```

Will be saved as a csv with the following columns: 

> **lab_id**: The identifier for each ex-vivo patient cell line. Should only be 5 id's (10250 is excluded). 

> **drugA**: One of the drug combinations.   

> **drugB**: Trametinib   

> **ICxx**: The XX value used to calculate the inhibitory constant, e.g., IC50 would be represented as ICxx=0.5  

> **min_CI**: The minimum CI value calculated at the given ICxx value. 

> **Ca_min**: The concentration of drug A at the combination concentration that results in `min_CI`   

> **Cb_min**:                  ... drug B  ...     

> **Ca_range_lower**: The smallest concentration of drug A at which the CI value is less than 1 (synergistic).   

> **Ca_range_upper**: The largest concentration of drug A at which the CI value is less than 1 (synergistic).   

> **Cb_range_lower**:                          ... drug B ...   

> **Cb_range_upper**:                          ... drug B ...   

> **ICxxA**: The calculated `ICxx` value of drug A. In units of log10( uM ). If the dose-response curve did not drop below `ICxx` within the measured dose range then it is assigned an arbitrarily large value (100 uM -> 2 log10(um)). 

> **ICxxB**:                            ... drug B ... 

> **equality_A**: The equality sign of ICxxA, can be either `equal_to` or `greater_than`   

> **equality_B**:                  ... ICxxB ...   