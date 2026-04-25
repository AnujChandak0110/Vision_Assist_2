import pyttsx3
import sys

def test_tts():
    try:
        engine = pyttsx3.init()
        print("Engine initialized")
        engine.setProperty('rate', 150)
        engine.say("Testing audio. If you can hear this, text to speech is working.")
        print("Running wait")
        engine.runAndWait()
        print("Done")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_tts()
