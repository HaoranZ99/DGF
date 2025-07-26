python src/main.py --dataset Movies --aas
python src/main.py --dataset Toys --aas
python src/main.py --dataset GroceryS --aas
python src/main.py --dataset RedditS --learning_rate 0.0001 --aas
# Larger datasets
python src/main.py --dataset Grocery --batch_size 5120 --aas --theta 0.1
python src/main.py --dataset Reddit --learning_rate 0.01 --batch_size 6144 --aas --theta 0.1
python src/main.py --dataset Photo --learning_rate 0.0001 --batch_size 10240 --aas --theta 0.1
python src/main.py --dataset Arts --batch_size 10240 --aas --theta 0.2