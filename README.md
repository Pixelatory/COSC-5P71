# COSC-5P71 Genetic Programming (GP)

## Assignment 1

### Part A

The purpose of part A was familiarize the student with GP, and to perform such a system for symbolic regression. A total of four experiments were completed with different parameter values, but a consistent function and terminal set.

My report for part A of assignment one is found [here](https://github.com/Pixelatory/COSC-5P71/blob/main/A1/COSC_5P71_A1_Part_A.pdf).

### Part B

Part B is a little different than part A, in that this is a classification task on a well-known breast cancer dataset (found [here](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)).

Since this data has 10 usable attributes for classification, and provides the mean, worst, and standard deviation of all attributes, I decided to inspect if all those values are even needed, or if a subset is sufficient. For my experiments, I selected all possible combinations of data points (all, mean, mean-worst, mean-st. dev., etc.), and showed how this may affect the results. This notions slightly towards feature selection, which is an approach closely tied with machine learning, where redundant attributes are eliminated from use to improve performance (Ref: [here](https://doi.org/10.1016/j.neucom.2017.11.077)).

The report for part B of assignment one is found [here](https://github.com/Pixelatory/COSC-5P71/blob/main/A1/COSC_5P71_A1_Part_B.pdf).

## Assignment 2

Assignment two is an image segmentation task. I decided to do such a task on boats, both docked and in open water. Various image filters are used in the function set (see the Pillow library [here](https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html) for many filters), and a pre-defined number of samples within the training section are used. As is expected, an increased number of samples leads to better performance. Offsets and RGB imagery was later introduced for the segmentation task, but only the latter showed improved performance, and on only one of the images. It was found that GP was exceptional at correctly classifying boats in open water, but performed poorly when boats were docked.

A version of SegNet was implemented as I planned to include this in my report (comparing GP against a modern segmentation framework), but lacked the time to do so.

The report for assignment two is found [here](https://github.com/Pixelatory/COSC-5P71/blob/main/A2/COSC_5P71_A2.pdf).

## Project

For my final project, I decided to try and improve upon the results achieved in assignment two, but with a clear focus on docked boats (as that's where the struggles lied). First, I tried a prioritized sampling technique. This was just taking samples directly from known dock pixels and using this during training. It showed that performance vastly improved doing this (in testing), even if the results from training looked bad. Additionally, I followed an intuition where just because it's hard to classify a boat with a nearby dock, it may not be hard to do the reverse. I attempted this, with some sharing mechanism to create an overall model for segmentation of docked boats. This method did not work well at all, and it was seen that it was harder to segment for a dock than for boats.

The report for the final project is found [here](https://github.com/Pixelatory/COSC-5P71/blob/main/Project/COSC_5P71_Final_Project.pdf).
