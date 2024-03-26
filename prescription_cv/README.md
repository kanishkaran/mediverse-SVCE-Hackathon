# Prescription Word Recognition Project

## Overview

This project aimed to develop a system to read doctors' prescriptions, initially exploring approaches using Convolutional Neural Networks (CNN) and Convolutional Recurrent Neural Networks (CRNN). Unfortunately, due to a lack of appropriate datasets and expertise in handling handwritten prescriptions, these attempts were unsuccessful. The `word_recog` directory contains the code for these failed approaches.

Ultimately, the project settled for an Optical Character Recognition (OCR) approach, specifically utilizing EasyOCR to extract text from digital prescriptions. While this approach doesn't address handwritten prescriptions directly, it provides a pragmatic solution given the constraints.

## Folders

### ocr

In the `ocr` folder, you'll find the implementation for reading digital prescriptions using EasyOCR. This module extracts characters from digital prescriptions and returns them as strings. The output can then be fed into a chatbot, which could potentially communicate with a backend SQL database to identify medicines (this integration is yet to be implemented).

### word_segmentation

The `word_segmentation` folder contains code for an attempted approach to segmenting words within images, which was abandoned early on due to limitations and the pursuit of better alternatives. Inspired by the repository at [githubharald/WordDetectorNN](https://github.com/githubharald/WordDetectorNN).

### word_recog

In `word_recog`, there are two main approaches explored for recognizing handwritten words. The first involves training models using the IAM dataset, but this failed to generalize to doctors' handwriting and lacked domain-specific knowledge. Relevant GitHub repositories consulted include:
- [SimpleHTR by githubharald](https://github.com/githubharald/SimpleHTR)
- [Handwritten-word-recognition-OCR----IAM-dataset---CNN-and-BiRNN by naveen-marthala](https://github.com/naveen-marthala/Handwritten-word-recognition-OCR----IAM-dataset---CNN-and-BiRNN/tree/master)
- [Handwritten-Text-Recognition by tuandoan998](https://github.com/tuandoan998/Handwritten-Text-Recognition)

The second approach in `word_recog` attempted to use object detection techniques but faced challenges such as underrepresented classes and a large number of classes (e.g., tablet names). This approach was inspired by the dataset available at [Roboflow](https://universe.roboflow.com/daffodil-international-university-s5vpr/merged-voyoh).

## Summary

This project serves as a learning experience, highlighting various attempts and challenges encountered in developing a prescription word recognition system. Due to time constraints, referenced resources were utilized extensively. The repository contains a curated collection of failed approaches alongside the implemented OCR solution for digital prescriptions.

