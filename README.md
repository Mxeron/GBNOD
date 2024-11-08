# GBNOD


## Usage
You can run GBNOD.py:
```
if __name__ == '__main__':
    data = pd.read_csv("./Example.csv").values
    sigma = 0.6
    n, m = data.shape
    OS = GBNOD(data, sigma)
    print(OS)
```
You can get outputs as follows:
```
OS = [0.36248495 0.38842417 0.33514463 0.38842417 0.38842417 0.34719267
 0.34719267 0.37670487 0.36568583 0.36248495 0.36248495 0.33514463
 0.36558735 0.33877366 0.37670487 0.40813498 0.36568583 0.33877366
 0.36568583 0.33877366]
```

## Contact
If you have any questions, please contact suxinyu@stu.scu.edu.cn.
