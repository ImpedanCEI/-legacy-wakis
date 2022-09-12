:file_folder: tests/
===

Contains the scripts that validate critical parts of the Wakis source code

:file_folder: fft/ 
---

Contains the test for the FFT function used in wakis to go from Wake potential to Impedance. It uses data from CST simulation of a simple cubic pillbox cavity. It reads the data of `Wp.txt` and `lambda.txt` and obtains the impedance by: 

![\Large Z_{||}(w) = -c \frac{\int_{-\infty}^{\infty}W_{||}(s)e^{-iws}ds }{\int_{-\infty}^{\infty} \lambda (s)e^{-iws}ds} 
}{2a}](https://latex.codecogs.com/png.latex?%5Cdpi%7B80%7D%20%5CLARGE%20Z_%7B%7C%7C%7D%28w%29%20%3D%20-c%20%5Cfrac%7B%5Cint_%7B-%5Cinfty%7D%5E%7B%5Cinfty%7DW_%7B%7C%7C%7D%28s%29e%5E%7B-iws%7Dds%20%7D%7B%5Cint_%7B-%5Cinfty%7D%5E%7B%5Cinfty%7D%20%5Clambda%20%28s%29e%5E%7B-iws%7Dds%7D)

Then compares the result to `Z.txt` file.


