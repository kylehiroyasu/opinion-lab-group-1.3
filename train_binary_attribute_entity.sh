python src/Main.py restaurants attribute -embedding glove -lr 0.0005 -cuda -b -use-kcl --binary-target-class GENERAL -s
python src/Main.py restaurants attribute -embedding glove -lr 0.0005 -cuda -b -use-kcl --binary-target-class MISCELLANEOUS -s
python src/Main.py restaurants attribute -embedding glove -lr 0.0005 -cuda -b -use-kcl --binary-target-class PRICES -s
python src/Main.py restaurants attribute -embedding glove -lr 0.0005 -cuda -b -use-kcl --binary-target-class QUALITY -s
python src/Main.py restaurants attribute -embedding glove -lr 0.0005 -cuda -b -use-kcl --binary-target-class STYLE_OPTIONS -s
python src/Main.py restaurants attribute -embedding glove -lr 0.0005 -cuda -b -use-kcl --binary-target-class NaN -s
python src/Main.py restaurants attribute -embedding bert-base-cased -lr 0.00005 -cuda -b -use-kcl --binary-target-class GENERAL -s
python src/Main.py restaurants attribute -embedding bert-base-cased -lr 0.00005 -cuda -b -use-kcl --binary-target-class MISCELLANEOUS -s
python src/Main.py restaurants attribute -embedding bert-base-cased -lr 0.00005 -cuda -b -use-kcl --binary-target-class PRICES -s
python src/Main.py restaurants attribute -embedding bert-base-cased -lr 0.00005 -cuda -b -use-kcl --binary-target-class QUALITY -s
python src/Main.py restaurants attribute -embedding bert-base-cased -lr 0.00005 -cuda -b -use-kcl --binary-target-class STYLE_OPTIONS -s
python src/Main.py restaurants attribute -embedding bert-base-cased -lr 0.00005 -cuda -b -use-kcl --binary-target-class NaN -s
python src/Main.py restaurants attribute -embedding glove -lr 0.0005 -cuda -b --binary-target-class GENERAL -s
python src/Main.py restaurants attribute -embedding glove -lr 0.0005 -cuda -b --binary-target-class MISCELLANEOUS -s
python src/Main.py restaurants attribute -embedding glove -lr 0.0005 -cuda -b --binary-target-class PRICES -s
python src/Main.py restaurants attribute -embedding glove -lr 0.0005 -cuda -b --binary-target-class QUALITY -s
python src/Main.py restaurants attribute -embedding glove -lr 0.0005 -cuda -b --binary-target-class STYLE_OPTIONS -s
python src/Main.py restaurants attribute -embedding glove -lr 0.0005 -cuda -b --binary-target-class NaN -s
python src/Main.py restaurants attribute -embedding bert-base-cased -lr 0.00005 -cuda -b --binary-target-class GENERAL -s
python src/Main.py restaurants attribute -embedding bert-base-cased -lr 0.00005 -cuda -b --binary-target-class MISCELLANEOUS -s
python src/Main.py restaurants attribute -embedding bert-base-cased -lr 0.00005 -cuda -b --binary-target-class PRICES -s
python src/Main.py restaurants attribute -embedding bert-base-cased -lr 0.00005 -cuda -b --binary-target-class QUALITY -s
python src/Main.py restaurants attribute -embedding bert-base-cased -lr 0.00005 -cuda -b --binary-target-class STYLE_OPTIONS -s
python src/Main.py restaurants attribute -embedding bert-base-uncased -cuda -b --binary-target-class NaN -s

python src/Main.py restaurants entity -embedding glove -cuda -s -b -use-kcl --binary-target-class AMBIENCE -lr 0.0005
python src/Main.py restaurants entity -embedding glove -cuda -s -b -use-kcl --binary-target-class DRINKS -lr 0.0005
python src/Main.py restaurants entity -embedding glove -cuda -s -b -use-kcl --binary-target-class FOOD -lr 0.0005
python src/Main.py restaurants entity -embedding glove -cuda -s -b -use-kcl --binary-target-class LOCATION -lr 0.0005
python src/Main.py restaurants entity -embedding glove -cuda -s -b -use-kcl --binary-target-class RESTAURANT -lr 0.0005
python src/Main.py restaurants entity -embedding glove -cuda -s -b -use-kcl --binary-target-class SERVICE -lr 0.0005
python src/Main.py restaurants entity -embedding glove -cuda -s -b -use-kcl --binary-target-class NaN -lr 0.0005
python src/Main.py restaurants entity -embedding glove -cuda -s -b --binary-target-class AMBIENCE -lr 0.0005
python src/Main.py restaurants entity -embedding glove -cuda -s -b --binary-target-class DRINKS -lr 0.0005
python src/Main.py restaurants entity -embedding glove -cuda -s -b --binary-target-class FOOD -lr 0.0005
python src/Main.py restaurants entity -embedding glove -cuda -s -b --binary-target-class LOCATION -lr 0.0005
python src/Main.py restaurants entity -embedding glove -cuda -s -b --binary-target-class RESTAURANT -lr 0.0005
python src/Main.py restaurants entity -embedding glove -cuda -s -b --binary-target-class SERVICE -lr 0.0005
python src/Main.py restaurants entity -embedding glove -cuda -s -b --binary-target-class NaN -lr 0.0005
python src/Main.py restaurants entity -embedding bert-base-cased -cuda -s -b -use-kcl --binary-target-class AMBIENCE -lr 0.00005
python src/Main.py restaurants entity -embedding bert-base-cased -cuda -s -b -use-kcl --binary-target-class DRINKS -lr 0.00005
python src/Main.py restaurants entity -embedding bert-base-cased -cuda -s -b -use-kcl --binary-target-class FOOD -lr 0.00005
python src/Main.py restaurants entity -embedding bert-base-cased -cuda -s -b -use-kcl --binary-target-class LOCATION -lr 0.00005
python src/Main.py restaurants entity -embedding bert-base-cased -cuda -s -b -use-kcl --binary-target-class RESTAURANT -lr 0.00005
python src/Main.py restaurants entity -embedding bert-base-cased -cuda -s -b -use-kcl --binary-target-class SERVICE -lr 0.00005
python src/Main.py restaurants entity -embedding bert-base-cased -cuda -s -b -use-kcl --binary-target-class NaN -lr 0.00005
python src/Main.py restaurants entity -embedding bert-base-cased -cuda -s -b --binary-target-class AMBIENCE -lr 0.00005
python src/Main.py restaurants entity -embedding bert-base-cased -cuda -s -b --binary-target-class DRINKS -lr 0.00005
python src/Main.py restaurants entity -embedding bert-base-cased -cuda -s -b --binary-target-class FOOD -lr 0.00005
python src/Main.py restaurants entity -embedding bert-base-cased -cuda -s -b --binary-target-class LOCATION -lr 0.00005
python src/Main.py restaurants entity -embedding bert-base-cased -cuda -s -b --binary-target-class RESTAURANT -lr 0.00005
python src/Main.py restaurants entity -embedding bert-base-cased -cuda -s -b --binary-target-class SERVICE -lr 0.00005
python src/Main.py restaurants entity -embedding bert-base-cased -cuda -s -b --binary-target-class NaN -lr 0.00005

