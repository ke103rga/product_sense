import pandas as pd
from typing import Union, List, Optional, Dict, Tuple, Callable
from itertools import product
from abc import ABC
import numpy as np

from ..eventframing.eventframe import EventFrame
from ..eventframing.cols_schema import EventFrameColsSchema
from ..utils.time_unit_period import TimeUnitPeriod
from .dinamic_metric_plotter import DinamicMetricPlotter as dmp


class _Metric(ABC):
    def __init__(self, formula: Callable, name: str, description: str = None):
        """
        Initializes a new instance of the _Metric class.

        Args:
            formula (Callable): The formula to calculate the metric.
            name (str): The name of the metric.
            description (str, optional): A description of the metric. Defaults to None.
        """
        self.formula = formula
        self.name = name
        self.description = description

    @staticmethod
    def get_unique_combinations(data: pd.DataFrame, hue_cols: Union[str, List[str]]) -> List[Dict]:
        """
        Returns a list of all combinations of unique values for the hue_cols fields in the data set.

        Args:
            data (pd.DataFrame): The input data.
            hue_cols (Union[str, List[str]]): The name of the column or a list of column names.

        Returns:
            List[Dict]: A list of combinations of unique field values.

        Raises:
            ValueError: If a column specified in `hue_cols` does not exist in the DataFrame.
        """
        if isinstance(hue_cols, str):
            hue_cols = [hue_cols]

        for col in hue_cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' does not exist in the DataFrame.")

        unique_values = [data[col].unique() for col in hue_cols]

        # Generate all unique combinations of values
        combinations = list(product(*unique_values))

        result = [{hue_cols[i]: combo[i] for i in range(len(hue_cols))} for combo in combinations]
        return result

    @staticmethod
    def filter_data_frame(data: pd.DataFrame, hue_cols_combo: Dict) -> pd.DataFrame:
        """
        Filters a DataFrame by combinations of unique values for hue_cols.

        Args:
            data (pd.DataFrame): The input data.
            hue_cols_combo (Dict): A dictionary of unique value combinations for hue_cols.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        query = ''
        for col, col_value in hue_cols_combo.items():
            if isinstance(col_value, str) or isinstance(col_value, np.datetime64) or isinstance(col_value,
                                                                                                pd.Timestamp):
                query += f"{col} == '{col_value}' & "
            else:
                query += f"{col} == {col_value} & "
        query = query[:-3]
        return data.query(query)

    # @staticmethod
    def _get_data(self, data: Union[pd.DataFrame, 'EventFrame']) -> pd.DataFrame:
        """
        Extracts the data from the input, whether it's a DataFrame or an EventFrame.

        Args:
            data (Union[pd.DataFrame, EventFrame]): The input data, which can be a DataFrame or an EventFrame.

        Returns:
            pd.DataFrame: A copy of the underlying DataFrame.

        Raises:
            TypeError: If the input data is not a DataFrame or an EventFrame.
        """
        if isinstance(data, EventFrame):
            return data.data.copy()
        if isinstance(data, pd.DataFrame):
            return data.copy()

        raise TypeError("data must be pd.DataFrame or EventFrame")


class MetricKPI(_Metric):
    def __init__(self,
                 formula: Callable[[Union[pd.DataFrame, EventFrame], Optional[EventFrameColsSchema], dict], float],
                 name: str, description: Optional[str] = None):
        """
       Initializes a new instance of the MetricKPI class.

       Args:
           formula (Callable): The formula to calculate the KPI.
           name (str): The name of the KPI.
           description (str, optional): A description of the KPI. Defaults to None.
       """
        super().__init__(formula, name, description)

    def _get_data_pivot_template(self, data: pd.DataFrame, hue_cols: List[str]) -> pd.DataFrame:
        """
        Generates a pivot table template with all combinations of unique values from hue_cols.

        This template ensures that the final result contains all possible combinations,
        even if some combinations are missing in the original data.

        Args:
            data (pd.DataFrame): The input DataFrame.
            hue_cols (List[str]): A list of column names to use for creating the pivot table.

        Returns:
            pd.DataFrame: A DataFrame representing the pivot table template.
        """
        unique_values = [data[col].unique() for col in hue_cols]

        # Create multi-index from all possible unique combinations
        index = pd.MultiIndex.from_product(unique_values, names=hue_cols)

        pivot_template = pd.DataFrame(index=index).reset_index()
        return pivot_template

    def compute_single_value(self, data: Union[pd.DataFrame, 'EventFrame'],
                             formula_kwargs: Optional[Dict] = None) -> float:
        """
        Computes the KPI for the entire dataset as a single value.

        Args:
            data (Union[pd.DataFrame, EventFrame]): The input data.
            formula_kwargs (Optional[Dict], optional): Keyword arguments to pass to the formula. Defaults to None.

        Returns:
            float: The computed KPI value.
        """
        data = super()._get_data(data)
        if formula_kwargs is None:
            formula_kwargs = dict()
        return self.formula(data, **formula_kwargs)

    def compute_splitted_values(self, data: Union[pd.DataFrame, 'EventFrame'],
                                hue_cols: Optional[Union[str, List[str]]] = None,
                                formula_kwargs: Optional[Dict] = None,
                                fillna_value: float = 0) -> pd.DataFrame | float:
        """
        Computes the KPI for subgroups of the data, split by the unique values in hue_cols.

        Args:
            data (Union[pd.DataFrame, EventFrame]): The input data.
            hue_cols (Optional[Union[str, List[str]]], optional): The column(s) to split the data by. Defaults to None.
            formula_kwargs (Optional[Dict], optional): Keyword arguments to pass to the formula. Defaults to None.
            fillna_value (float, optional): The value to fill NaN values with after the computation. Defaults to 0.

        Returns:
            pd.DataFrame: A DataFrame with the computed KPI values for each subgroup.

        """
        data = super()._get_data(data)
        if formula_kwargs is None:
            formula_kwargs = dict()

        if hue_cols is None or len(hue_cols) == 0:
            return self.compute_single_value(data, formula_kwargs)

        if isinstance(hue_cols, str):
            hue_cols = [hue_cols]

        result = data.groupby(hue_cols) \
            .apply(lambda group_data: self.formula(group_data, **formula_kwargs)) \
            .reset_index().rename(columns={0: self.name})

        pivot_template = self._get_data_pivot_template(data, hue_cols)

        result = pd.merge(
            pivot_template,
            result,
            on=hue_cols,
            how='left'
        )
        result[self.name] = result[self.name].fillna(fillna_value)

        return result


class MetricDinamicComputeParams:
    def __init__(self, data: Union[pd.DataFrame, 'EventFrame'],
                 period: Union[str, TimeUnitPeriod],
                 hue_cols: Union[str, List[str]],
                 dt_col: str,
                 fillna_value: float,
                 formula_kwargs: Optional[Dict]) -> None:

        self.data, self.dt_col = self._get_data_and_dt_col(data, dt_col)

        if isinstance(period, str):
            period = TimeUnitPeriod(period)
        if formula_kwargs is None:
            formula_kwargs = dict()

        self.period = period
        self.period_name = period.alias
        self.hue_cols = hue_cols
        self.fillna_value = fillna_value
        self.formula_kwargs = formula_kwargs

    def _get_data_and_dt_col(self,
                             data: Union[pd.DataFrame, 'EventFrame'],
                             dt_col: Optional[str]) -> Tuple[pd.DataFrame, str]:

        if isinstance(data, EventFrame):
            dt_col = data.cols_schema.event_timestamp
            data = data.data.copy()
        elif isinstance(data, pd.DataFrame):
            if dt_col is None:
                raise ValueError('It\'s necessary to specify "dt_col" if data is pd.DataFrame')
        else:
            raise TypeError("data must be pd.DataFrame or EventFrame")

        return data, dt_col


class MetricDinamic(_Metric):
    """
    A class for computing dynamic metrics over time periods.

    Inherits from the _Metric class.

    Attributes:
        formula (Callable): The formula to calculate the dynamic metric. It should accept a DataFrame or EventFrame
        and optional keyword arguments.
        name (str): The name of the dynamic metric.
        description (str, optional): A description of the dynamic metric. Defaults to an empty string.
        compute_params (Optional[MetricDinamicComputeParams]): Parameters used for computing the dynamic metric.
        Defaults to None.
    """
    def __init__(self, formula: Callable[[Union[pd.DataFrame, EventFrame], dict], float],
                 name: str, description: str = ''):
        """
        Initializes a new instance of the MetricDinamic class.

        Args:
            formula (Callable): The formula to calculate the dynamic metric.
            name (str): The name of the dynamic metric.
            description (str, optional): A description of the dynamic metric. Defaults to an empty string.
        """
        super().__init__(formula, name, description)
        self.compute_params: Optional[MetricDinamicComputeParams] = None

    def _get_data_pivot_template(self, data: pd.DataFrame, dt_col: str,
                                 period: TimeUnitPeriod, hue_cols: List[str]) -> pd.DataFrame:
        """
        Generates a pivot table template with all combinations of time periods and hue_cols.

        This template ensures that the final result contains all possible combinations,
        even if some combinations are missing in the original data.

        Args:
            data (pd.DataFrame): The input DataFrame.
            dt_col (str): The name of the datetime column.
            period (TimeUnitPeriod): The time unit period for grouping data.
            hue_cols (List[str]): A list of column names to use for creating the pivot table.

        Returns:
            pd.DataFrame: A DataFrame representing the pivot table template.
        """
        min_date, max_date = data[dt_col].min(), data[dt_col].max()
        pivot_template = period.generte_monotic_time_range(min_date, max_date)
        if len(hue_cols) > 0:
            for col_name in hue_cols:
                col_values = data[col_name].unique()
                pivot_template = pd.merge(
                    pivot_template,
                    pd.DataFrame({col_name: col_values}),
                    how='cross'
                )
        return pivot_template

    def compute(self, data: Union[pd.DataFrame, 'EventFrame'],
                period: Union[str, TimeUnitPeriod] = 'D',
                hue_cols: Union[str, List[str]] = None,
                dt_col: str = None,
                fillna_value: float = 0,
                formula_kwargs: Optional[Dict] = None) -> pd.DataFrame:
        """
        Computes the dynamic metric for different time periods and subgroups of the data.

        Args:
            data (Union[pd.DataFrame, EventFrame]): The input data.
            period (Union[str, TimeUnitPeriod], optional): The time unit period for grouping data. Defaults to 'D' (day).
            hue_cols (Union[str, List[str]], optional): The column(s) to split the data by. Defaults to None.
            dt_col (str, optional): The name of the datetime column. Defaults to None.
            fillna_value (float, optional): The value to fill NaN values with after the computation. Defaults to 0.
            formula_kwargs (Optional[Dict], optional): Keyword arguments to pass to the formula. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame with the computed dynamic metric values for each time period and subgroup.
        """
        self.compute_params = MetricDinamicComputeParams(data, period, hue_cols, dt_col, fillna_value, formula_kwargs)

        data = self.compute_params.data
        dt_col = self.compute_params.dt_col
        period = self.compute_params.period
        period_name = self.compute_params.period_name
        formula_kwargs = self.compute_params.formula_kwargs

        data = period.add_period_col(data, dt_col, new_col_name=period_name)

        if hue_cols is None or len(hue_cols) == 0:
            hue_cols = []
        elif isinstance(hue_cols, str):
            hue_cols = [hue_cols]
        result = data.groupby([period_name] + hue_cols) \
            .apply(lambda group_data: self.formula(group_data, **formula_kwargs)) \
            .reset_index().rename(columns={0: self.name})

        pivot_template = self._get_data_pivot_template(data, dt_col, period, hue_cols)

        result = pd.merge(
            pivot_template,
            result,
            on=hue_cols + [period_name],
            how='left'
        )
        result[self.name] = result[self.name].fillna(fillna_value)
        return result.sort_values([period_name] + hue_cols)

    def plot(self, data: Union[pd.DataFrame, 'EventFrame'],
             period: Union[str, TimeUnitPeriod] = 'D',
             hue_cols: Union[str, List[str]] = None,
             dt_col: str = None,
             fillna_value: float = 0,
             formula_kwargs: Optional[Dict] = None,
             smooth: int = 0,
             engine: str = 'plotly',
             figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plots the dynamic metric data.

        Args:
            data (Union[pd.DataFrame, EventFrame]): The input data.
            period (Union[str, TimeUnitPeriod], optional): The time unit period for grouping data. Defaults to 'D' (day).
            hue_cols (Union[str, List[str]], optional): The column(s) to split the data by. Defaults to None.
            dt_col (str, optional): The name of the datetime column. Defaults to None.
            fillna_value (float, optional): The value to fill NaN values with after the computation. Defaults to 0.
            formula_kwargs (Optional[Dict], optional): Keyword arguments to pass to the formula. Defaults to None.
            smooth (int, optional): The smoothing factor for the plot. Defaults to 0.
            engine (str, optional): The plotting engine to use. Defaults to 'plotly'.
            figsize (Tuple[int, int], optional): The figure size for the plot. Defaults to (12, 6).
        """
        metric_data = self.compute(
            data=data,
            period=period,
            hue_cols=hue_cols,
            dt_col=dt_col,
            fillna_value=fillna_value,
            formula_kwargs=formula_kwargs,
        )

        new_dt_col = self.compute_params.period_name
        dmp.plot(
            metric_data,
            dt_col=new_dt_col,
            value_col=self.name,
            hue_cols=hue_cols,
            smooth=smooth,
            engine=engine,
            figsize=figsize
        )

    def plot_analyze(self, data: Union[pd.DataFrame, 'EventFrame'],
                     period: Union[str, TimeUnitPeriod] = 'D',
                     dt_col: str = None,
                     fillna_value: float = 0,
                     formula_kwargs: Optional[Dict] = None,
                     window_sizes: Optional[List[int]] = None,
                     lags: Optional[int] = None,
                     trend_line: bool = False,
                     figsize: Optional[Tuple[int, int]] = None) -> None:
        """
       Performs and plots an analysis of the dynamic metric data.

       Args:
           data (Union[pd.DataFrame, EventFrame]): The input data.
           period (Union[str, TimeUnitPeriod], optional): The time unit period for grouping data. Defaults to 'D' (day).
           dt_col (str, optional): The name of the datetime column. Defaults to None.
           fillna_value (float, optional): The value to fill NaN values with after the computation. Defaults to 0.
           formula_kwargs (Optional[Dict], optional): Keyword arguments to pass to the formula. Defaults to None.
           window_sizes (Optional[List[int]], optional): The window sizes for moving average calculation.
           Defaults to None.
           lags (Optional[int], optional): The number of lags for autocorrelation analysis. Defaults to None.
           trend_line (bool, optional): Whether to plot a trend line. Defaults to False.
        """

        metric_data = self.compute(
            data=data,
            period=period,
            hue_cols=None,  # It's possible to analyze only single time series
            dt_col=dt_col,
            fillna_value=fillna_value,
            formula_kwargs=formula_kwargs,
        )

        new_dt_col = self.compute_params.period_name

        dmp.plot_analysis(
            data=metric_data,
            dt_col=new_dt_col,
            value_col=self.name,
            window_sizes=window_sizes,
            lags=lags,
            trend_line=trend_line,
            figsize=figsize
        )
