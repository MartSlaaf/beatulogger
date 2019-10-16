from collections import defaultdict
from .history import History
from .canvas import Canvas
import numpy as np

import torch
import re
from numbers import Number

from tqdm import tqdm

class BeautyLogger:
    #TODO: add comments
    #TODO: add overridable aggregators?
    #TODO: add tests

    def __init__(self, aggregable=None, calculable=None, plots=None, progressbar='none', prints=None, print_mode='last', trackable=None,
                 tracking_mode=None, plot_backend='canvas', tb_parameters=None):
        """
        Class for logging of training process parameters. Could also aggregate metrics, plot or print them for you.
        Args:
            aggregable (list): parameters to be aggregated. First should be name of parameter, second should be either 'mean', 'max' or callable to get aggregate.
            calculable (list): parameters to be aggregated via complex function. Every list element should be tuple: (list of input parameters names, output parameter name, aggregating function)
            plots (list): plots to be shown. Every list element should be a tuple: (type of plot, list of parameters to plot)
            progressbar (str): could be either 'none' for no progress bar at all, 'epochs' for progress over epochs, 'steps' for progress bar over steps or 'both' for both steps and epochs progressbars.
            prints (list): parameters to be printed. Each element should be either string (parameter name) or pair (parameter name, mode: 'max'/'min'). If mode setted, maximum or minimum achieved value will be printed
            trackable (str): parameter name to track for early stopping or model saving. Is required for functions is_best and steps_without_progress
        """
        # setup aggregable parameters
        self.aggregable = {}
        if aggregable is not None:
            for aggregation_params in aggregable:
                self.add_aggregable(*aggregation_params)

        # setup calculable parameters
        self.calculable = []
        self.calculable_inputs = []
        if calculable is not None:
            for calculable_parameters in calculable:
                self.add_calculable(*calculable_parameters)

        # setup plots
        self.plots = []
        if plots is not None:
            for plot_definition in plots:
                self.add_plot(plot_definition)

        # setup tracking mode
        self.trackable = trackable
        self.tracking_mode = np.max if tracking_mode == 'max' else np.min

        # setup intenals
        self.inter_epoch = defaultdict(lambda:defaultdict(list))

        self.epochs = History()
        self.plot_backend = plot_backend
        if self.plot_backend == 'canvas':
            self.canvas = Canvas()
        elif self.plot_backend == 'tensorboard':
           global tb
           from torch.utils import tensorboard as tb
           self.writer_parameters = {} if tb_parameters is None else tb_parameters
        else:
            raise Exception('Unexpected plotting backend!')

        self.step = 0

        # progress-bars TBD:
        if prints is not None:
            self.prints = self._initialize_prints(prints)
        else:
            self.prints = None
        self.prin_mode = print_mode
        self.epochs_progressbar = None
        self.steps_progressbar = None
        if progressbar in ['epochs', 'both']:
            self.epochs_progressbar = tqdm()
        if progressbar in ['steps', 'both']:
            self.steps_progressbar = tqdm()

    def add_aggregable(self, metric_name, aggregation_type='mean'):
        """
        Adds agregable metrics to track.
        Args:
            metric_name (str): the metric name without parenthesis.
            aggregation_type (str, callable): the function to reduce iterations over epoch. Strings supported now are 'mean' and 'max'
        """
        if aggregation_type == 'mean':
            self.aggregable[metric_name] = np.mean
        elif aggregation_type == 'max':
            self.aggregable[metric_name] = np.max
        elif callable(aggregation_type):
            self.aggregable[metric_name] = agg_type
        else:
            raise ValueError('aggregation type expected to be "mean", "max" or callable.')

    def add_calculable(self, input_names, output_name, function):
        """
        Adds calculable metrics to track.
        Args:
            input_names (list): names (without parenthesis) of metrics to be aggregated into new one, in order of input to function.
            output_name (str): name of new metric to create after aggregation.
            function (callable): function to aggregate metrics.
        """
        self.calculable_inputs += input_names
        if callable(function):
            self.calculable.append((input_names, output_name, function))
        else:
            raise ValueError('Function expected to be callable')

    def add_plot(self, names_to_plot):
        """
        Adds one plot to track.
        Args:
            names_to_plot (str, list): name of parameter to plot, or list of names.
                If passed with parenthesis will plot selected type of iteration (train-test-etc), if without -- all available types for the same name.
        """
        if isinstance(names_to_plot, str):
            self.plots.append([names_to_plot])
        elif isinstance(names_to_plot, list):
            self.plots.append(names_to_plot)
        else:
            raise ValueError(f'metrics to plot {names_to_plot} are of incorrect type. Either str or list expected.')

    def _initialize_prints(self, prints):
        new_prints = []
        for element in prints:
            if isinstance(element, tuple):
                if element[1] == 'max':
                    new_prints.append((element[0], np.max))
                elif element[1] == 'min':
                    new_prints.append((element[0], np.min))
                else:
                    raise ValueError(f'Unknown printing mode {element[1]}')
            else:
                new_prints.append((element, None))

    def _get_value(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        else:
            return value

    def log_step(self, step_type='train', **kwargs):
        for param, value in kwargs.items():
            self.inter_epoch[step_type][param].append(self._get_value(value))

    def _concat_param(self, param_tile):
        if isinstance(param_tile[0], np.ndarray):
            return np.concatenate(param_tile, 0)
        elif isinstance(param_tile[0], Number):
            return np.array(param_tile)
        else:
            raise ValueError(f'Unknown type of parameter values {type(param_tile[0])}')

    def is_best(self, trackable=None, tracking_mode=None):
        if (self.trackable is None) and (trackable is None):
            raise Exception('Best epoch could be estimated only with setted trackable parameter. Set it on initialization or pass to this function.')
        if (self.tracking_mode is None) and (tracking_mode is None):
            raise Exception('Best epoch could be estimated only with setted tracking mode. Set it on initialization or pass to this function.')
        else:
            if tracking_mode is not None:
                tracking_mode = np.max if tracking_mode == 'max' else np.min
            else:
                tracking_mode = self.tracking_mode
        trackable = trackable if trackable is not None else self.trackable
        track = self.epochs[trackable].data
        return track[-1] == tracking_mode(track)

    def steps_without_progress(self, trackable=None, tracking_mode=None):
        if (self.trackable is None) and (trackable is None):
            raise Exception('Steps without progress could be estimated only with setted trackable parameter. Set it on initialization or pass to this function.')
        if (self.tracking_mode is None) and (tracking_mode is None):
            raise Exception('Steps without progress could be estimated only with setted tracking mode. Set it on initialization or pass to this function.')
        else:
            if tracking_mode is not None:
                tracking_mode = np.max if tracking_mode == 'max' else np.min
            else:
                tracking_mode = self.tracking_mode
        trackable = trackable if trackable is not None else self.trackable

        track = self.epochs[trackable].data
        best_value = tracking_mode(track)
        return len(track) - (np.where(track==best_value)[0][-1] + 1)

    def _concat_params(self, step_type, param_names):
        return [self._concat_param(self.inter_epoch[step_type][par_n]) for par_n in param_names]

    def agg_epoch(self, step_type='train'):
        # build temporal aggregables_list:
        tmp_aggregable = dict()
        for param in self.inter_epoch[step_type].keys():
            if param in self.aggregable.keys():
                tmp_aggregable[param] = self.aggregable[param]
            elif (param not in self.calculable_inputs):
                tmp_aggregable[param] = 'mean'

        # aggregate all params meant in aggregable
        # TODO: move definition to init
        if tmp_aggregable:
            agg_funcs, agg_params = [], []
            for param, agg_type in tmp_aggregable.items():
                if agg_type == 'mean':
                    func_to_agg = np.mean
                elif agg_type == 'max':
                    func_to_agg = np.max
                elif callable(agg_type):
                    func_to_agg = agg_type
                else:
                    raise ValueError('aggregation type expected to be "mean", "max" or callable')
                agg_funcs.append(func_to_agg)
                agg_params.append(param)

            self.epochs.log(self.step, **{n+'('+step_type+')': f(p) for p,f,n in zip(self._concat_params(step_type, agg_params), agg_funcs, agg_params)})

        if self.calculable is not None:
            for input_params, output_param, convert_function in self.calculable:
                self.epochs.log(self.step, **{output_param+'('+step_type+')': convert_function(*self._concat_params(step_type, input_params))})

    def log_epoch(self, **kwargs):
        for step_type in self.inter_epoch.keys():
            self.agg_epoch(step_type)

        self.epochs.log(self.step, **kwargs)

        self.step += 1
        self.inter_epoch = defaultdict(lambda:defaultdict(list))

    def plot(self):
        if self.plot_backend == 'canvas':
            self.plot_canvas()
        elif self.plot_backend == 'tensorboard':
            self.plot_tensorboard()

    def plot_tensorboard(self):
        writer = tb.SummaryWriter(**self.writer_parameters)
        for plot_elements in self.plots:
            if isinstance(plot_elements, str):
                plot_elements = [plot_elements]
            new_plot_elements = []
            for plot_element in plot_elements:
                if '(' in plot_element:
                    new_plot_elements.append(plot_element)
                else:
                    new_plot_elements += sorted([elemname for elemname in self.epochs.metrics if re.match(f'^{plot_element}(\(.+?\))?$', elemname)])

            writer.add_scalars(new_plot_elements[0], {p_e: self.epochs[p_e].data[-1] for p_e in new_plot_elements}, global_step=self.step)
        writer.close()

    def plot_canvas(self):
        with self.canvas:
            for plot_elements in self.plots:
                if isinstance(plot_elements, str):
                    plot_elements = [plot_elements]
                new_plot_elements = []
                for plot_element in plot_elements:
                    if '(' in plot_element:
                        new_plot_elements.append(plot_element)
                    else:
                        new_plot_elements += sorted([elemname for elemname in self.epochs.metrics if re.match(f'^{elemname}(\(.+?\))?$', elemname)])
                self.canvas.draw_plot([self.epochs[p_e] for p_e in new_plot_elements])

    def print(self):
        if self.print_mode == 'last':
            string_to_write = ''
            for param, modifier in self.prints:
                value = self.epochs[param].data[-1]
                if modifier is not None:
                    best_value = modifier(self.epochs[param].data)
                    string_to_write += f'{value:.3} ({best_value:.3})\t'
                else:
                    string_to_write += f'{value:.3}\t'

            if self.epochs_progressbar is not None:
                self.epochs_progressbar.write(string_to_write, end='\r')
            elif self.steps_progressbar is not None:
                self.steps_progressbar.write(string_to_write, end='\r')
            else:
                print(string_to_write, end='\r')
        elif self.print_mode == 'all':
            pass
        elif self.print_mode == 'exponential':
            pass
