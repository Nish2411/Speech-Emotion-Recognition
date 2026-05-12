import sounddevice as sd 
import soundfile
import pickle

from utils import extract_feature, TARGET_SAMPLE_RATE

# Load trained model
model = pickle.load(open("model/emotion_model.pkl","rb"))

# record audio

def record_audio(filename, duration=3, sample_rate=TARGET_SAMPLE_RATE):
    print("\nSpeak now...")

    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    soundfile.write(filename, recording.squeeze(), sample_rate)
    print("Recording complete!")

#predict emotion

def predict_file(filename):
    features = extract_feature(filename)
    if features is None:
        return None, None

    features = features.reshape(1, -1)

    prediction = model.predict(features)[0]
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = dict(zip(model.classes_, model.predict_proba(features)[0]))
    return prediction, probabilities


def predict_emotion():
    filename = "temp.wav"

    #record voice
    record_audio(filename)

    prediction, probabilities = predict_file(filename)
    if prediction is None:
        print("\nCould not detect enough voice in the recording. Please try again.")
        return

    print(f"\nPredicted Emotion: {prediction}")
    if probabilities is not None:
        print("Confidence:", {label: round(float(score), 3) for label, score in probabilities.items()})


def main():
    print("\n============Speech Emotion Recognition===========")

    while True:
        predict_emotion()
        again = input("\nTry again? (y/n): ")
        if again.lower() != 'y':
            break
    print("\nProgram ended.")


if __name__ == "__main__":
    main()
