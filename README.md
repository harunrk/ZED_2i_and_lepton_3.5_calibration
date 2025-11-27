# ZED 2i & Lepton 3.5 Calibration

This project integrates the **ZED 2i RGB-D camera** and the **FLIR Lepton 3.5 thermal camera** to capture synchronized data and perform full calibration between the two sensors.

🧠 **How It Works**

The system records RGB-D and thermal frames **at the same time**.  
Both cameras are calibrated individually, and then a **stereo calibration** is performed to compute the spatial alignment between RGB and thermal images.

📌 **Use Cases**

  - Multispectral sensing  
  - Robotics & autonomous navigation  
  - Thermal + RGB fusion  
  - Research experiments & prototyping  

📷 **Data Included**

  - Synchronized RGB + thermal frames  
  - Calibration checkerboard images  
  - Thermal calibration datasets  
  - Intrinsic & extrinsic calibration files  

⚙️ **Requirements**

  - ZED SDK + ZED Python API  
  - OpenCV  
  - `flirpy` or similar Lepton libraries  
  - Python 3.x
