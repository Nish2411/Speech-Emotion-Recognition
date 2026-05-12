import os
import glob
import numpy as np 
import pickle

from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from utils import extract_feature

#emotion mapping

emotions = {
    '01':'neutral',
    '02':'happy',
    '03':'sad',
    '04':'angry',

}

# Emotions we want

observed_emotions = ['neutral','happy','sad','angry']

# Load dataset

def load_data():
    x, y, groups = [], [], []

    #path to dataset
    for file in glob.glob("dataset/**/*.wav", recursive=True):
        file_name = os.path.basename(file)
        parts = file_name.split("-")

        #emotion code from filename
        if len(parts) < 7:
            continue

        emotion_code = parts[2]
        actor_id = parts[6].split(".")[0]

        emotion = emotions.get(emotion_code)
        
        if emotion in observed_emotions: 
            feature = extract_feature(file)
            if feature is not None:
                x.append(feature)
                y.append(emotion)
                groups.append(actor_id)
    return np.array(x), y, np.array(groups)

# Load features and Labels

print("loading dataset and extracting features...")
X, y, groups = load_data()

print(f"Total samples: {len(X)}")

#split dataset by speaker

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

train_idx, test_idx = next(splitter.split(X, y, groups))
x_train, x_test = X[train_idx], X[test_idx]
y = np.array(y)
y_train, y_test = y[train_idx], y[test_idx]
train_actors = sorted(set(groups[train_idx]))
test_actors = sorted(set(groups[test_idx]))
print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))
print("Training actors:", ", ".join(train_actors))
print("Testing actors:", ", ".join(test_actors))

# create MLP Model

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", MLPClassifier(
        alpha=0.01,
        batch_size=256,
        epsilon=1e-08,
        hidden_layer_sizes=(300,),
        learning_rate='adaptive',
        max_iter=500,
        random_state=42
    ))
])

# train model

print("\nTraining model.....")
model.fit(x_train,y_train)
print("Training completed!")


# predictions

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print("\nAccuracy:", round(accuracy*100,2),"%")

print("\nClassification Report:\n")
print(classification_report(y_test,y_pred))

os.makedirs("model", exist_ok=True)

with open("model/emotion_model.pkl","wb") as f:
    pickle.dump(model,f)
print("\nModel saved successfully!")    
