import pyttsx3
import multiprocessing
import time

def worker(q):
    try:
        engine = pyttsx3.init()
        while True:
            text = q.get()
            if text is None:
                break
            print("Speaking:", text)
            engine.say(text)
            engine.runAndWait()
    except Exception as e:
        print("Error in worker:", e)

if __name__ == "__main__":
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=worker, args=(q,))
    p.start()
    
    q.put("Testing multiprocessing audio.")
    time.sleep(2)
    q.put(None)
    p.join()
    print("Done")
