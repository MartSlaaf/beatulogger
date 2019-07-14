# BeautyLogger

A very lightweight library for online tracking of your experiments. It is useful for those cases when you whant to track progress right in your notebook.

This is fork of perfectly simple library [HiddenLayer](https://github.com/waleedka/hiddenlayer) by Waleed Abdulla and Phil Ferriere.

## Usage Scenario
The main idea is to put all the metrics together, and free user of aggregating and plotting metrics by himself.
Usually when you whant to calculate AUROC you should log your prediction and labels at every step, than finally convert it, calculate it and may be also check how much epochs it stays without progress.
The target of BeautyLogger is to equip you with tunable logger which will aggregate, check and plot for you.

The workflow looks as follows:
1. Before training cycle, you setup the logger parameters. 
Which metrics should we aggregate in some original way, which pre-requisites should be passed to some function to became new metrics. 
What plots do you want to observe.
1. At the each step you pass values of the metrics (e.g. loss-components) and pre-requisites for metrics to be calculated epoch-wise (e.g. prediction and label for AUROC).
1. At the end of epoch you call function which would aggregate epoch data pass (You could pass additional epoch-wise metrics, if you have.)
1. If you want to update information on the screen, you just call `.plot()` method.
If you want to check if this current epoch best according to predefined metrics you call another function, `.is_best()`.
If you want to check how much epochs there was without progress corresponding to predefined metrics -- `.steps_without_progress()`

## NB!
All the logging variables at log-time should not contain parenthesis, because parenthesis right now are used for coding of step type (e.g. train or test).

## Usage
The plug-n-play interface of BeautyLogger have a number of simple parameters:
1. `aggregable` expected to be list of tuples. First element of each tuple is name of metric and second is string or callable. Callable should expect numpy array as input and return float. Strings could be 'mean' or 'max'. 
1. `calculable` expected to be list of tuples. First element of each tuple is list of names of metrics to be inputs of an aggregator function. Second element is string containing name for new parameter -- result of calculations. Third is callable -- function to aggregate respected parameters. So if you pass tuple (a, b, c) expected result is b = c(\*a)
1. `plots` expected to be list of strings and lists. String is tracted as the name of metric to plot. List should contain number of strings -- names of metrics to plot on the same plot. All names without parenthesis will trigger plotting all variants (train, test or valid)
1. `trackable` -- string with full name (with parenthesis) of parameter for tracking. This parameter will drive early stopping and best model saving.
1. `tracking_mode` should be either 'min' or 'max'. It defines is the trackable parameter lesser-better or bigger-better.

## Examples
Simple example of usage on fake data could be found [here](demos/fake.ipynb) 

