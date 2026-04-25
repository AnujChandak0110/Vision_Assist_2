# Setting Up an External Webcam

Follow these steps to switch from your built-in camera to an external USB webcam for the Vision Assist project.

## 1. Connect Your Webcam
Plug your external USB webcam into an available USB port on your computer. Make sure any drivers are installed if required by the manufacturer.

## 2. Identify the Camera Index
Windows assigns a numeric "index" to each connected camera. Usually, `0` is the built-in laptop camera, and `1` (or higher) is the external webcam.

We have created a helper script to find the exact index of your newly connected webcam:
1. Open your terminal in the `Vision_Assist_2` project directory.
2. Run the detection script:
   ```bash
   python detect_cameras.py
   ```
3. The output will look something like this:
   ```
   Scanning for cameras (index 0-5)...

     [0] FOUND — resolution=640x480  fps=0  *** USE THIS INDEX ***
     [1] FOUND — resolution=640x480  fps=0  *** USE THIS INDEX ***

   Available camera indexes: [0, 1]
   ```
4. Note the index number of the camera you want to use (likely `1`).

## 3. Configure the `.env` File
Now that the system is updated to support environment-based configuration, you do not need to modify any Python code.

1. Open the `.env` file in the root of the project directory.
2. Add or update the following line with the index number you found in the previous step:
   ```env
   CAMERA_INDEX=1
   ```
   *(Replace `1` with your actual camera index).*

## 4. Run the Application
Restart the main application. It will automatically load the new `CAMERA_INDEX` and connect to your external webcam using the optimized Windows DirectShow backend for faster and more stable performance.
```bash
python main.py
```
