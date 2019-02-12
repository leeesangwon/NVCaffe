1. Download and extract bdd100k dataset. By default, we assume the data is stored in `$HOME/data/`

2. Create the train.txt, val.txt, and val_name_size.txt in `data/bdd100k/`
  ```Shell
  cd $CAFFE_ROOT/datda/bdd100k
  python ./create_list.py
  ```

3. Create the LMDB file.
  ```Shell
  # You can modify the parameters in create_data.sh if needed.
  # It will create lmdb files for train and val with encoded original image:
  #   - $HOME/data/BDD/bdd100k/bdd100k/lmdb/bdd100k_train_lmdb
  #   - $HOME/data/BDD/bdd100k/bdd100k/lmdb/bdd100k_val_lmdb
  # and make soft links at examples/bdd100k/
  cd $CAFFE_ROOT
  ./data/bdd100k/create_data.sh
  ```