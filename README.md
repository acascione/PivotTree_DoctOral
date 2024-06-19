# PivotTree_DoctOral

For feature extraction and respective feature dataframe creation, it is needed:

- `oral_datas/` folder including `./data/images/` which store oral lesion images
- `oral_datas/` folder which includes `train.json`,`test.json`,`val.json` files which include annotations information and reference to respective 

`Doctoral_NB.ipynb` specifies the extraction process


For the model assessment, a folder named `data` must be avilable containing the dataset containing feature emebedding from the Detectron2 architecture:

- `oral2_train_features.json`
- `oral2_test_features.json`
- `oral2_val_features.json`



`PT_validation.py` is used to generate validation/testing results with PivotTree, both as a classifier and as a prototype selection tool

A folder named `dataset_stz` must be avilable containing a folder `dataset_stz/splits`, respectively containing
- `dataset_stz/splits/training_sets`
- `dataset_stz/splits/validation_sets`
- `dataset_stz/splits/test_sets`

Each folder hosts the training, validation and testing splits from the dataset taken into account. Therefore, given a dataset named `df_oral.csv`, we will have in the respective folders `train_df_oral.csv`, `validation_df_oral.csv`, `test_df_oral.csv`, where `df_oral.csv` stands for the name of the full dataset.





