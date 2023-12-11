# useful_tools

### file path extraction
```
python3 file_path_extractor.py \
-s folder/to/get/files \
-d folder/to/save/results \
-sfn file/name \
-end_with file/type
```

### Similarity Calculation
```
python3 imp_gen_fast.py \
-p probe/feature/file \
-g gallery/feature/file \
-o output/folder \
-d dataset/name \
-gr group/name
```
### Identity Disjoint for Assigning Pairs to Folds
```
python3 assign_pairs_to_folds.py \
-gen genuine/pair/file \
-imp impostor/pair/file \
-d output/folder \
-name output/file/name
```
Note that this algorithm only considers the first identity of each impostor pair, 
but it is still able to provide a good result. If you can come up with a better algorithm, welcome to share
it. :)