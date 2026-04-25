import win32com.client
speaker = win32com.client.Dispatch("SAPI.SpVoice")
speaker.Speak("Testing raw SAPI5 engine directly.")
print("SAPI5 spoken.")
