# Introduction to Machine Learning and Deep Learning in Python (2-Day Course)

## By [Irina Chelysheva](https://github.com/Chelysheva) ([Oxford Profile](https://www.brookes.ac.uk/profiles/staff/irina-chelysheva)) and [Srinivasa Rao Rao](https://github.com/sraorao) ([Oxford Profile](https://www.nds.ox.ac.uk/team/srinivasa-rao))

## Course Aim

This course provides a practical introduction to applying machine learning and deep learning algorithms to biological data using the Python programming language. Through a blend of lectures and hands-on training, participants will learn the steps of data processing, preparation, and application of machine learning techniques to solve various problems, guided by expert tutors. By the end of the course, participants should be able to apply machine learning and deep learning techniques to both tabular and image data.

### Session Details

You are expected to attend two half-day sessions:

- **Tuesday, 20th October**: 10:00 - 13:00
- **Friday, 23rd October**: 10:00 - 13:00

## Requirements

### Day 1: machine learning

Python 3.10 or newer is recommended. Use Spyder, Jupyter, VS Code or another Python editor.

```bash
python -m pip install -r requirements_day1.txt
```

Keep these files together:

- `ML_Python_day1_classification.py`
- `breast_cancer_wisconsin_original.csv`
- `ML_Python_day1_regression.py`
- `diabetes_regression.csv`

Run the scripts in numbered sections in Spyder, or from a terminal:

```bash
python ML_Python_day1_classification.py
python ML_Python_day1_regression.py
```

With Jupyter, use `%run ML_Python_day1_classification.py` or `%run ML_Python_day1_regression.py`. If pasting individual sections into notebook cells, set the working directory to the folder containing the CSV files.

Figures, cross-validation summaries, held-out predictions and fitted pipelines are saved in `outputs_classification/` and `outputs_regression/`. These folders are generated outputs and do not need to be committed. Figures are also displayed at the end of each script. Only load joblib model files you trust.

Day 1 slides: `Session_1_ML_intro_python_Oct2026.pptx`. Optional backup slides are hidden in slide-show view but remain accessible in the editor. See `DATASETS.md` for provenance and a data dictionary.

### Day 2: deep learning

The separate Day 2 scripts, slides and `deeplearning.yaml` are unchanged. If you have conda/mamba, create the supplied deep-learning environment:

```bash
conda env create -f deeplearning.yaml
conda activate deeplearning
```

## Course Objectives

By the end of this course, you will learn:

- How to pre-process your data for machine learning and deep learning analysis.
- How to apply machine learning techniques to address biological questions using example datasets.
- How to apply deep learning on an example image dataset.
- How to evaluate and visualize analysis results.

## Course Format

The course is structured into two half-day sessions as follows:

### Day 1

**Theory:**
- Introduction to Machine Learning
- Types of Problems That Can Be Resolved with Machine Learning
- Overview of Machine Learning Algorithms
- Feature Selection Algorithms

**Hands-on 1:**
- Application of machine learning in Python using the `sklearn` library on the Wisconsin Breast Cancer Original dataset (classification, with repeated sample IDs kept together during validation).

**Hands-on 2:**
- Application of feature selection and machine learning in Python using the `sklearn` library on the diabetes progression dataset (regression: ten baseline predictors and a progression score at one year).

### Day 2

**Theory:**
- Introduction to Image Analysis with Deep Learning
- Use Cases
- Image Pre-processing

**Hands-on:**
- Pre-processing images in Python using `NumPy` and `OpenCV` libraries.

**Theory:**
- Convolutional Neural Networks (CNNs) using `PyTorch`

**Hands-on:**
- Application of CNNs on an image classification problem (Acute Lymphoblastic Leukemia classification).
- Model Evaluation
- Using Pre-trained Models
