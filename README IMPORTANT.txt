The dataset used can be downloaded here - https://www.kaggle.com/code/sandhyakrishnan02/face-mask-detection-using-pytorch/data

This dataset consists of 4095 images belonging to two classes:

with_mask: 1105 images
without_mask.xml: 880 images
The images used were real images of faces wearing masks. 


To create a .NET 6 class library that loads an image and the trained model to infer whether a person in the image is wearing a mask or not, follow these steps:

1-Create a new .NET 6 class library project in Visual Studio.

2-Add the TensorFlow.NET NuGet package to the project. This package provides a C# wrapper for TensorFlow and makes it easy to load and use a pre-trained model.

3-Create a class that represents the image and its metadata. This class should have a property that represents the path to the image file.

4-Create a class that represents the TensorFlow model and its metadata. 
This class should have a property that represents the path to the model file.

5-Create a class that contains a method for inferring whether the person in the image is wearing a mask or not.
 This method should use TensorFlow.NET to load the model and the image, and then use the model to make a prediction.
 The prediction can be a boolean value that indicates whether the person in the image is wearing a mask or not.

6-Build the project and create a NuGet package for the class library.
