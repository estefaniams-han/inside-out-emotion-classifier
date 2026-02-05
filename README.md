# Inside Out Emotion Classifier

An emotion classifier based on the Inside Out movies that analyzes text and predicts one of 7 main emotions: anger, disgust, fear, sadness, joy, surprise, and neutral.

<p align="center">
<img width="763" height="357" alt="image" src="https://github.com/user-attachments/assets/901228a0-b2d2-4fd4-8b7b-1c706129a76c" />
</p>

## Description

This project uses natural language processing (NLP) to classify text according to the main emotions from the Inside Out movie. The model is trained on Google Research's **GoEmotions** dataset, which contains Reddit texts annotated with 28 different emotions, mapped to the 7 Inside Out categories.

## Motivation

This project was created as a personal learning initiative to deepen my understanding of machine learning and natural language processing. By working hands-on with real datasets, experimenting with different models, and building an end-to-end ML application, I aim to strengthen my practical skills in NLP and ML engineering.

## Features

- Text classification into 7 Inside Out emotions
- Displays probabilities for each emotion
- Interactive web application with Streamlit
- Machine learning model based on TF-IDF + Logistic Regression
- Optimization with GridSearchCV for best results

## Project Structure

```
mood-stress-nlp/
├── app/
│   └── streamlit_app.py       # Interactive web application
├── data/
│   ├── raw/                   # Raw data
│   └── processed/             # Processed data
├── models/
│   └── emotion_lr_tfidf_best.joblib  # Trained model
├── notebooks/
│   └── 01_eda_go_emotions.ipynb      # Exploratory data analysis
├── src/
│   └── train_emotion.py       # Training script
└── requirements.txt           # Project dependencies
```

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd mood-stress-nlp
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Run the web application

```bash
streamlit run app/streamlit_app.py
```

The application will open in your default browser at `http://localhost:8501`

### Train the model

If you want to retrain the model with different parameters:

```bash
python src/train_emotion.py
```

This script:
1. Automatically downloads the GoEmotions dataset
2. Maps the 28 original emotions to the 7 Inside Out categories
3. Trains a model using GridSearchCV
4. Evaluates performance on the validation set
5. Saves the best model to `models/`

## Emotion Mapping

The model maps GoEmotions to Inside Out emotions as follows:

| Inside Out | GoEmotions |
|-----------|------------|
| 😡 Anger | anger, annoyance |
| 🤢 Disgust | disgust, disapproval, contempt |
| 😱 Fear | fear, nervousness |
| 😢 Sadness | sadness, grief, disappointment, remorse |
| 😄 Joy | joy, amusement, love, excitement |
| 😲 Surprise | surprise, realization |
| 😐 Neutral | neutral (and other unmapped emotions) |

## Model Details

- **Algorithm**: Logistic Regression with TF-IDF
- **Dataset**: GoEmotions (~43,410 training examples)
- **Validation**: Stratified train-test split (80/20)
- **Optimization**: GridSearchCV with macro F1-score
- **Features**:
  - N-grams: (1,1) or (1,2)
  - TF-IDF with sublinear transformation
  - Custom class weights to balance classes
  - L2 regularization

## Dataset

This project uses the [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) dataset created by Google Research, which contains:

- 58,000+ Reddit comments
- 28 emotion categories
- Multi-label annotations

## Technologies Used

- **Python 3.11**
- **Streamlit** - Interactive web interface
- **scikit-learn** - Machine learning model
- **Hugging Face Datasets** - Download and manage GoEmotions dataset
- **pandas** & **numpy** - Data manipulation
- **matplotlib** - Visualizations (in notebooks)

## Future Improvements

### Planned Enhancements

- Test word n-grams (1,2) with best parameters (min_df=5, max_df=0.9, sublinear_tf=True)
- Experiment with character n-grams (3,5) as an alternative model
- If disgust/fear remain weak → implement two-stage classification (neutral vs emotion first)
- Explore advanced models: LinearSVC or BERT-based transformers


## License

This project uses the GoEmotions dataset which is licensed under Apache License 2.0.

## Author

Estefania Marmolejo

## Acknowledgments

- Google Research for the GoEmotions dataset
- Pixar for the Inside Out inspiration
