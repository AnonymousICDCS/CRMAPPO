# CRMAPPO Implementation

This repository provides a simple and clean implementation of CRMAPPO.

## Usage

You can run the program using the following command format in Python:

```bash
python main.py --beta=0.6 --user_num=6 --seed=1 --write=True
```

Alternatively, you can directly run the included .sh files. The results will be saved in TensorBoard.

Different algorithms are organized into separate directories for clarity. The non-RL algorithm (i.e., the Average) is not included here, as it can be easily implemented by modifying the listed code. The traditional MAPPO is not provided separately, as CRMAPPO is essentially MAPPO when $beta=0.

## Implementation Details
+ Environments for all algorithms are consistent.
+ Parameters can be conveniently adjusted in the main.py files for various algorithms.
  
Feel free to explore the code and tailor it to your specific needs.

## Some Important Packges
- Python 3.8.13
- gym 0.25.0
- torch 1.13.0
