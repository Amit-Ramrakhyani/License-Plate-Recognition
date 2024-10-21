# License Plate Detection and Recognition

This project implements a license plate detection and recognition system. 

## Images

To run the image prediction app, follow the steps below:

1. Clone the repository

```bash
git clone https://github.com/Amit-Ramrakhyani/License-Plate-Recognition.git
```

2. Change to the project directory
```bash
cd License-Plate-Recognition
```

3. Install the required packages using the following command:
```bash
python -m pip install -r requirements.txt
```

**NOTE**: You also need to install "Tesseract-OCR" for the OCR part in `recognition-tesseractocr.ipynb`

For Ubuntu/Debian based systems, you can install it using the following commands:

```bash
sudo apt update
sudo apt install tesseract-ocr
```

1. Go to the app directory
```bash
cd app
```

1. This project is dependent on the **Sort** module. You can download it using the following command: 

```bash
git clone https://github.com/abewley/sort.git
```

6. Run the following command to start the application:
```bash
streamlit run app.py
```

The app will open in your default browser. You can also access it by going to http://localhost:8501 in your browser.

## Videos

To run the video analysis part, keep the video in the `/images/input` directory and change the path of the video in the `/app/main.py` & `/app/visualize.py` files. Then run the following commands in order:

```bash
python3 main.py
```

```bash
python3 add_missing_data.py
```

```bash
python3 visualize.py
```

The output video will be available in the `/images/output/output.mp4`

**NOTE**: Running a large video file without a GPU might be the reason your system crashs. That's why the computation is made to be done on only 500 frames of the video. You can change that on the 29th line of `/app/main.py`. You're welcome ;)