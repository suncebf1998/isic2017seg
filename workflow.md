# Work Flow of ISICSeg

## Work Classes Define

1. class TODO and TODOList: TODOList is the container of TODOs
   1. has "status" attribution: in ("0", "1", "-1", "2")
      1. "0": just make TODO
      2. "1": done
      3. "-1": deprecated
      4. “2”: unfinished
   2. has "result" attribution: str
2. class Time: has "deadline","startime","delta" attribution
   1. "deadline","startime": yyyymmdd
   2. "delta" Optional;: xx.y \[hours\]
3. class Classification: has "level","details" attribution
   1. "level": in ("finding","reading","coding","running","writing")
   2. "details": describe the level, for example -- finding + keyword or View; reading + papername; coding + aliasname + ("debug" + "merge" + "create" + "update" + "recreate_struc" + "recreate_error" + "recreate_beautify" + ...)
4. class Sync: has "hub","repository", "msg" attribution
   1. "msg": yyyymmdd Classification.level + Classification.detials

## Work Notes

### June
#### Day 10
```python
Job2022_06_10_01(TODO, Time, Classification):
    TODO:
        status: 0
    Time: 
        startime: 20220610 
        deadline: 20220610 
        delta: 1
    Classification: 
        level: coding
        details: MultiAveragePool update
    """
    torch.cat -> other
    map 20220610
    instead cat, use add: reduce some memory used
    Total   307809
    Trainable       307809
    Running Total Time: 3820.20 seconds A100 40GB
    """
```

```python
Job2022_06_10_02(TODO, Time, Classification):
    TODO:
        status: 0
    Time: 
        startime: 20220610 
        deadline: 20220610 
        delta: 1
    Classification: 
        level: coding
        details: MultiAveragePool recreate_struc
    """
    Cs + MAPs -> (C + MAP)s
    Total   89409
    Trainable       89409
    8 in
    info: Saved model weights.                                                                                                                          
    Running Total Time: 2562.88 seconds 

    Total   408833
    Trainable       408833
    info: Saved model weights.                                                                                                                          
    Running Total Time: 4234.59 seconds    


    f1 is at 0.85 far from unet 0.91 and swinunet 0.93
    """
```

```python
Job2022_06_10_03(TODO, Time, Classification):
    TODO:
        status: 0
    Time: 
        startime: 20220610 
        deadline: 20220610 
        delta: None
    Classification: 
        level: running
        details: Job2022_06_10_02
    """
    Cs + MAPs -> (C + MAP)s
    """
```

```python
Job2022_06_10_00(TODOList, Time, Sync):
    TODOList:
        status: 0
        result: map之间无明显区别在Adam中极限在0.9左右，swinunet不能证明强于unet
        container: Job2022_06_10_01 : Job2022_06_10_03
    Time: 
        startime: 20220610 
        deadline: 20220610 
        delta: None
    Sync: 
        hub: git
        repository: isic2017seg.git
        msg: 20220610 MultiAveragePool update and recreate_struc

    """
    Cs + MAPs -> (C + MAP)s
    """
```
----unet----31M
Total   31 037 633
Trainable       31 037 633
info: Saved model weights.                                                                                                                         
Running Total Time: 3980.60 seconds   

----swinunet----19M
Total   19 344 516
Trainable       19 344 516
info: Saved model weights.                                                                                                                         
Running Total Time: 3576.46 seconds    

----map----0.408M
Total   408 833
Trainable       408 833
info: Saved model weights.                                                                                                                         
Running Total Time: 4301.94 seconds  


#### Day 13 - 14
```python
Job2022_06_1314_01(TODO, Time, Classification):
    TODO:
        status: 0
    Time: 
        startime: 20220613 
        deadline: 20220614
        delta: 2
    Classification: 
        level: coding
        details: create UperNet
    """
    https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation
    https://arxiv.org/abs/1807.10221
    """

Job2022_06_1314_02(TODO, Time, Classification):
    TODO:
        status: 0
    Time: 
        startime: 20220614 
        deadline: 20220614
        delta: 1
    Classification: 
        level: running
        details: UperNet

Job2022_06_1314_00(TODOList, Time, Sync):
    TODOList:
        status: 0
        result: upernet参数和swinunet相近，但效果可能优于 swinunet 和 unet
        container: Job2022_06_1314_01 : Job2022_06_1314_03
    Time: 
        startime: 20220613 
        deadline: 20220614 
        delta: None
    Sync: 
        hub: git
        repository: isic2017seg.git
        msg: 20220610 MultiAveragePool update and recreate_struc
```

```
    ----upernet----
    Total   19 424 609
    Trainable       19 424 609
    Running Total Time: 4322.06 seconds
````


```python
Job2022_06_16_01(TODO, Time, Classification):
    TODO:
        status: 0
    Time: 
        startime: 20220616 
        deadline: 20220616
        delta: 1
    Classification: 
        level: coding
        details: create swin unet laterals
    """
    https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation
    https://arxiv.org/abs/1807.10221
    """
```

```
Job2022_06_16_00(TODOList, Time, Sync):
    TODOList:
        status: 0
        result: upernet参数和swinunet相近，但效果可能优于 swinunet 和 unet
        container: Job2022_06_1314_01 : Job2022_06_1314_03
    Time: 
        startime: 20220616
        deadline: 20220616 
        delta: None
    Sync: 
        hub: git
        repository: isic2017seg.git
        msg: 20220616 coding create swin unet lateral v1 and recreate swinunet without softmax
```

```
    ----swinlateral----
    Total   21 331 652
    Trainable       21 331 652
                                                                                                                        
    Running Total Time: 2310.13 seconds  

    ----swin unet without softmax V2----
    Total   19344516
    Trainable       19344516
                                                                                                                                   
    Running Total Time: 2113.26 seconds   
```