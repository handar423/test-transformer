### INSTALL
```shell
git clone -b horovod https://github.com/handar423/test-transformer.git
cd test-transformer
pip install -e .
pip install sklearn
python utils/download_glue_data.py --data_dir ~/data/glue --tasks all
```