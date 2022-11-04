# Code for Task3
We utilize the DKT model for this task. This DKT model can track the students' knowledge status through the students' writing records, then, we generate the submission data by students' knowledge status. 
One need to perform the following steps:
- Process the raw training data for the DKT model.
- Process the output of the DKT model for submission.

For this program, we recommend using a video card with a video memory of at least 10GB. For us, we use the GTX 1080 (12GB) graphics card to run this program. You can also run this program using the CPU, but for a longer time.

The data directory stores the data used in the competition

All the following steps require setting up the necessary dependencies using poetry.  
## Process the raw training data for the DKT model
```bash
python datapre.py
```
This will output 1 files: part_cons_train.csv.

## Process the output of the DKT model fro submission
```bash
cd DKT
```
If you have the trained models, you only execute. Or, you need to run 'python run.py' to generate the model.
```bash
python tiqu.py
```
This will output 1 files: storage

Then
```bash
cd data
python trans.py
cd ../..
```

Last
```bash
python submission.py
```
The submitted file will be generated in the result directory
