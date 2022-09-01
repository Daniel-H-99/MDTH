# Mesh-Driving Talking Head

Contact: **Daniel**

## Prepare

### Installation
Set `prefix` in environment.yaml
``` shell
conda env create -f environment.yaml
conda activate MDTH
```
Install `ps-body` package as in [psbody-github](https://github.com/MPI-IS/mesh)

---

### Load pre-trained checkpoints
Required checkpoint / config / static files are in
`odinson:/mnt/aitrics_ext/ext01/warping-shared/warping-common/th`

Sample input data is prepared in S3/raw/th_test (Even though there are multiple audio / image / video as input, only one of them is filter during preprocessing)

Input name / format are not forced since they are adapted during preprocessing

## Instruction

### Structures

+ Storage
```
/raw
+-- {twin_id}
    +-- audio
    |   +-- input audio file
    +-- image
    |   +-- input image file
    +-- video
        +-- input video file
/processed
+-- {twin_id}
    +-- preprocessed
            +-- th
                +-- audio
                |   +-- procesesd audio file
                +-- image
                |   +-- processed image related file
                +-- video
                    +-- processed video related file
/inference
+-- {twin_id}
    +-- th
        +-- output files
```

+ Local
```
/warping_shared
+-- warping-common
|   +-- th
|       +-- checkpoints
|       +-- flame
+-- warping-data
|   +-- {twin_id}
|       +-- audio
|       |   +-- downloaded audio file
|       +-- image
|       |   +-- downloade image file
|       +-- video
|           +-- downloaded video file
+-- warping-processed
|   +-- {twin_id}
|       +-- processed
|           +-- th
|               +-- audio
|               |   +-- procesesd audio file
|               +-- image
|               |   +-- processed image related file
|               +-- video
|                   +-- processed video related file
+-- warping-serving
    +-- {twin_id}
        +-- th
            +-- output files
```

### Code Structure
`*/export.py`: include all exporting functions from the module
`BFv2v`: Module for Head Warping
`FaceFormer`: Module Audio 2 Mesh

```
/TH
+-- BFv2v
    +-- export.py
    +-- ...
+-- FaceFormer
    +-- export.py
    +-- ...
+-- pipeline
```
### Test
By using `guarantee`, 

+ Preprocess
```
python run_guarantee.py --model TH --task preprocess --twin-id th_test --n-gpus 1
```
+ Inference
```
python run_guarantee.py --model TH --task inference --twin-id th_test --n-gpus 1
```
