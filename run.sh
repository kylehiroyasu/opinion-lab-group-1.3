python src/Main.py restaurants attribute -embedding bert-base-cased -lr 0.00005 -cuda -b --binary-target-class GENERAL -e 1000 -use-kcl
python src/Main.py restaurants attribute -embedding bert-base-cased -lr 0.00005 -cuda -b --binary-target-class MISCELLANEOUS -e 1000 -use-kcl
python src/Main.py restaurants attribute -embedding bert-base-cased -lr 0.00005 -cuda -b --binary-target-class PRICES -e 1000 -use-kcl
python src/Main.py restaurants attribute -embedding bert-base-cased -lr 0.00005 -cuda -b --binary-target-class QUALITY -e 1000 -use-kcl
python src/Main.py restaurants attribute -embedding bert-base-cased -lr 0.00005 -cuda -b --binary-target-class STYLE_OPTIONS -e 1000 -use-kcl
python src/Main.py restaurants attribute -embedding bert-base-cased -lr 0.00005 -cuda -b --binary-target-class NaN -e 1000 -use-kcl

python src/Main.py restaurants entity -embedding bert-base-cased -cuda -b --binary-target-class AMBIENCE -lr 0.00005 -e 1000 -use-kcl
python src/Main.py restaurants entity -embedding bert-base-cased -cuda -b --binary-target-class DRINKS -lr 0.00005 -e 1000 -use-kcl
python src/Main.py restaurants entity -embedding bert-base-cased -cuda -b --binary-target-class FOOD -lr 0.00005 -e 1000 -use-kcl
python src/Main.py restaurants entity -embedding bert-base-cased -cuda -b --binary-target-class LOCATION -lr 0.00005 -e 1000 -use-kcl
python src/Main.py restaurants entity -embedding bert-base-cased -cuda -b --binary-target-class RESTAURANT -lr 0.00005 -e 1000 -use-kcl
python src/Main.py restaurants entity -embedding bert-base-cased -cuda -b --binary-target-class SERVICE -lr 0.00005 -e 1000 -use-kcl
python src/Main.py restaurants entity -embedding bert-base-cased -cuda -b --binary-target-class NaN -lr 0.00005 -e 1000 -use-kcl

