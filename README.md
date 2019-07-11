# BeautyLogger

A very lightweight library for online tracking of your experiments. It is useful for those cases when you whant to track progress right in your notebook.

This is fork of perfectly simple library [HiddenLayer](https://github.com/waleedka/hiddenlayer) by Waleed Abdulla and Phil Ferriere.

## Usage
The plug-n-play interface of BeautyLogger have a number of simple parameters:
1. `aggregable` expected to be list of tuples. First element of each tuple is name of metric and second is string or callable. Callable should expect numpy array as input and return float. Strings could be 'mean' or 'max'. 
1. `calculable` expected to be list of tuples. First element of each tuple is list of names of metrics to be inputs of an aggregator function. Second element is string containing name for new parameter -- result of calculations. Third is callable -- function to aggregate respected parameters. So if you pass tuple (a, b, c) expected result is b = c(\*a)
1. `plots` expected to be list of strings and lists. String is tracted as the name of metric to plot. List should contain number of strings -- names of metrics to plot on the same plot. All names without parenthesis will trigger plotting all variants (train, test or valid)
1. `trackable` -- string with full name (with parenthesis) of parameter for tracking. This parameter will drive early stopping and best model saving.
1. `tracking_mode` should be either 'min' or 'max'. It defines is the trackable parameter lesser-better or bigger-better.

## Examples
Simple example of usage on fake data could be found [here](demos/fake.ipynb) 

