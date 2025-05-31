from typing import Literal, Union, Optional, get_args, Dict
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from ...eventframing.eventframe import EventFrame
from ...metrics.metric import MetricKPI
from ...data_preprocessing.preprocessors_lib.add_cohorts_preprocessor import AddCohortsPreprocessor
from ...utils.time_unit_period import TimeUnitPeriod


class Cohorts:
    """A class for analyzing and visualizing cohort data.

    This class provides methods to create, analyze and visualize cohort tables
    based on different KPIs and time periods.
    """
    RepresentationTypes = Literal['time_unit', 'period']

    def __init__(self):
        self.kpi_metric = None
        self.cohort_period = None
        self.represent_by = None
        self.cohort_table = None
        self.normalize = False

    def _prepare_fit_data(self, data: EventFrame, extract_cohorts: bool, cohort_period: Union[TimeUnitPeriod, str]):
        """Prepares data for cohort analysis.

        Args:
            data: EventFrame containing the data to analyze.
            extract_cohorts: Whether to extract cohorts from the data.
            cohort_period: Time period for cohort grouping.

        Returns:
            Tuple containing the prepared data and column schema.
        """
        if extract_cohorts:
            cohorts_preprocessor = AddCohortsPreprocessor(cohort_period)
            data = cohorts_preprocessor.apply(data)

        cols_schema = data.cols_schema
        data = data.data.copy()

        return data, cols_schema

    def fit(self, data: EventFrame,
            extract_cohorts: bool = True,
            cohort_period: Union[TimeUnitPeriod, str] = 'D',
            represent_by: RepresentationTypes = 'time_unit',
            normalize: bool = False) -> pd.DataFrame:
        """Fits the cohort model using unique user counts as the metric.

        Args:
            data: EventFrame containing user event data.
            extract_cohorts: Whether to extract cohorts from the data.
            cohort_period: Time period for cohort grouping.
            represent_by: How to represent cohorts ('time_unit' or 'period').
            normalize: Whether to normalize the cohort table.

        Returns:
            DataFrame containing the cohort analysis results.
        """
        if isinstance(cohort_period, str):
            cohort_period = TimeUnitPeriod(cohort_period)

        data, cols_schema = self._prepare_fit_data(data, extract_cohorts, cohort_period)

        cohort_col = cols_schema.cohort_group
        represent_by_col = 'cohort_time_unit' if represent_by == 'time_unit' else 'cohort_period'
        user_id_col = cols_schema.user_id

        self.cohort_period = cohort_period
        self.represent_by = represent_by
        self.normalize = normalize

        cohort_table = data.pivot_table(
            index=cohort_col,
            columns=represent_by_col,
            values=user_id_col,
            aggfunc='nunique'
        )

        cohort_table = cohort_table.reset_index().melt(id_vars=['cohort_group'])
        if represent_by == 'time_unit':
            cohort_table['cohort_time_unit'] = pd.to_datetime(cohort_table['cohort_time_unit'])
        cohort_table = self._expand_cohort_table(
            cohort_table=cohort_table,
            represent_by=represent_by,
            represent_by_col=represent_by_col,
            values_col='value'
        )

        if normalize:
            cohort_table = self.normalize_cohort_table(cohort_table, represent_by)

        self.cohort_table = cohort_table
        return cohort_table

    def fit_by_custom_kpi(self, data: EventFrame,
                          kpi_metric: MetricKPI,
                          kpi_metric_kwargs: Optional[Dict] = None,
                          extract_cohorts: bool = True,
                          cohort_period: Union[TimeUnitPeriod, str] = 'D',
                          represent_by: RepresentationTypes = 'time_unit',
                          normalize: bool = False) -> pd.DataFrame:
        """Fits the cohort model using a custom KPI metric.

        Args:
            data: EventFrame containing user event data.
            kpi_metric: MetricKPI instance to use for cohort analysis.
            kpi_metric_kwargs: Additional arguments for the KPI metric.
            extract_cohorts: Whether to extract cohorts from the data.
            cohort_period: Time period for cohort grouping.
            represent_by: How to represent cohorts ('time_unit' or 'period').
            normalize: Whether to normalize the cohort table.

        Returns:
            DataFrame containing the cohort analysis results.
        """

        self._check_fit_params(data, represent_by)

        if isinstance(cohort_period, str):
            cohort_period = TimeUnitPeriod(cohort_period)

        data, cols_schema = self._prepare_fit_data(data, extract_cohorts, cohort_period)

        cohort_col = cols_schema.cohort_group
        represent_by_col = 'cohort_time_unit' if represent_by == 'time_unit' else 'cohort_period'

        self.cohort_period = cohort_period
        self.represent_by = represent_by
        self.kpi_metric = kpi_metric

        cohort_table = kpi_metric.compute_splitted_values(
            # data=data.astype(str),
            data=data,
            hue_cols=[cohort_col, represent_by_col],
            formula_kwargs=kpi_metric_kwargs
        )
        cohort_table[cohort_col] = pd.to_datetime(cohort_table[cohort_col])
        if represent_by == 'time_unit':
            cohort_table[represent_by_col] = pd.to_datetime(cohort_table[represent_by_col])
        else:
            cohort_table[represent_by_col] = cohort_table[represent_by_col].astype(int)

        cohort_table = self._expand_cohort_table(
            cohort_table=cohort_table,
            represent_by=represent_by,
            represent_by_col=represent_by_col,
            values_col=kpi_metric.name
        )

        if normalize:
            cohort_table = self.normalize_cohort_table(cohort_table, represent_by)
        self.normalize = normalize

        self.cohort_table = cohort_table
        return cohort_table

    def normalize_cohort_table(self, cohort_table: pd.DataFrame, represent_by: RepresentationTypes) -> pd.DataFrame:
        """Normalizes the cohort table values.

        Args:
            cohort_table: DataFrame containing cohort data.
            represent_by: How cohorts are represented ('time_unit' or 'period').

        Returns:
            Normalized cohort table.
        """
        if represent_by == 'time_unit':
            cohort_table = cohort_table.divide(np.diag(cohort_table), axis=0)
        else:
            cohort_table = cohort_table.divide(cohort_table.iloc[:, 0], axis=0)
        return cohort_table

    def plot(self, annot: bool = True, fmt=None, annot_kws=None, cmap=None, title=None,
             min_cohort: Optional[Union[str, pd.Timestamp]] = None,
             max_cohort: Optional[Union[str, pd.Timestamp]] = None,
             min_period: Optional[int] = None, max_period: Optional[int] = None,
             min_time_unit: Optional[Union[str, pd.Timestamp]] = None,
             max_time_unit: Optional[Union[str, pd.Timestamp]] = None) -> None:
        """Visualizes the cohort table as a heatmap.

        Args:
            annot: Whether to annotate heatmap cells with values.
            fmt: Format string for annotations.
            annot_kws: Keyword arguments for annotations.
            cmap: Colormap for the heatmap.
            title: Title for the plot.
            min_cohort: Minimum cohort to include.
            max_cohort: Maximum cohort to include.
            min_period: Minimum period to include.
            max_period: Maximum period to include.
            min_time_unit: Minimum time unit to include.
            max_time_unit: Maximum time unit to include.
        """
        # cohort_table = self.cohort_table.copy()
        cohort_table = self._prepare_cohort_table(
            min_cohort, max_cohort, min_period, 
            max_period, min_time_unit, max_time_unit
        )

        cohort_table.set_index(cohort_table.index.astype(str), inplace=True)
        cohort_table.columns = cohort_table.columns.astype(str)

        if fmt is None:
            fmt = '.0%' if self.normalize else '.0f'
        if annot_kws is None:
            annot_kws = {'fontsize': 10}
        if cmap is None:
            cmap = sns.color_palette("light:b", as_cmap=True)
        if title is None:
            metric_name = self.kpi_metric.name if self.kpi_metric is not None else 'Unique users'
            title = f'{metric_name} by {self.cohort_period.alias} cohorts'

        if self.represent_by == 'time_unit':
            ylabel = self.cohort_period.alias
        else:
            ylabel = f'{self.cohort_period.alias}(s) after first visit'

        fig, axes = plt.subplots(figsize=(16, 6))
        sns.heatmap(data=cohort_table, mask=cohort_table.isnull(),
                    annot=annot, fmt=fmt, ax=axes, annot_kws=annot_kws, cmap=cmap)
        axes.set_title(title)
        axes.set_xlabel(ylabel)
        axes.set_ylabel("Cohort group")
        plt.tight_layout()

    @property
    def values(self):
        """Returns a copy of the cohort table.

        Returns:
            DataFrame containing cohort analysis results.
        """
        return self.cohort_table.copy()

    def _prepare_cohort_table(self,
                              min_cohort: Optional[Union[str, pd.Timestamp]] = None,
                              max_cohort: Optional[Union[str, pd.Timestamp]] = None,
                              min_period: Optional[int] = None, max_period: Optional[int] = None,
                              min_time_unit: Optional[Union[str, pd.Timestamp]] = None,
                              max_time_unit: Optional[Union[str, pd.Timestamp]] = None,
                              group_mean: bool = False, period_mean: bool = False) -> pd.DataFrame:
        """Prepares cohort table for visualization with optional filtering.

        Args:
            min_cohort: Minimum cohort to include.
            max_cohort: Maximum cohort to include.
            min_period: Minimum period to include.
            max_period: Maximum period to include.
            min_time_unit: Minimum time unit to include.
            max_time_unit: Maximum time unit to include.
            group_mean: Whether to add cohort group means.
            period_mean: Whether to add period means.

        Returns:
            Filtered and processed cohort table.
        """

        cohort_table = self.cohort_table.copy()

        min_cohort = min_cohort if min_cohort is not None else min(cohort_table.index)
        max_cohort = max_cohort if max_cohort is not None else max(cohort_table.index)
        cohorts = cohort_table.index
        cohorts = cohorts[(cohorts >= min_cohort) & (cohorts <= max_cohort)]
        if (len(cohorts) == 0):
            raise ValueError(f'No cohorts which are between {str(min_cohort)} and {str(max_cohort)}')
        cohort_table = cohort_table.loc[cohorts]

        if self.represent_by == 'time_unit':
            min_time_unit = np.datetime64(min_time_unit) if min_time_unit is not None else min(cohort_table.columns)
            max_time_unit = np.datetime64(max_time_unit) if max_time_unit is not None else max(cohort_table.columns)
            cols = (np.datetime64(col) for col in cohort_table.columns if col >= min_time_unit and col <= max_time_unit)
            cols = tuple(cols)
            if (len(list(cols)) == 0):
                raise ValueError(f'No time units which are between {str(min_time_unit)} and {str(max_time_unit)}')

        if self.represent_by == 'period':
            min_period = int(min_period) if min_period is not None else min(cohort_table.columns)
            max_period = int(max_period) if max_period is not None else max(cohort_table.columns)
            cols = (col for col in cohort_table.columns if col >= min_period and col <= max_period)
            cols = tuple(cols)

            if (len(list(cols)) == 0):
                raise ValueError(f'No cohort periods which are between {str(min_period)} and {str(max_period)}')
        cohort_table = cohort_table.loc[:, tuple(cols)]

        if group_mean:
            cohort_table['cohort_group_mean'] = cohort_table.mean(axis=1)
        if period_mean:
            cohort_table.loc['period_mean'] = cohort_table.mean(axis=0)

        return cohort_table

    def _prepare_pivot_template(self, cohorts_data: pd.DataFrame, represent_by: str) -> pd.DataFrame:
        """Creates a template for pivoting cohort data.

        Args:
            cohorts_data: Raw cohort data.
            represent_by: How cohorts are represented ('time_unit' or 'period').

        Returns:
            DataFrame template for pivoting.
        """
        time_unit_name = self.cohort_period.alias
        cohort_group_min, cohort_group_max = cohorts_data['cohort_group'].min(), cohorts_data['cohort_group'].max()
        cohort_group_monotic_range = self.cohort_period.generte_monotic_time_range(cohort_group_min, cohort_group_max) \
            .rename(columns={time_unit_name: 'cohort_group'})

        if represent_by == 'time_unit':
            cohort_tu_min, cohort_tu_max = (cohorts_data['cohort_time_unit'].min(),
                                            cohorts_data['cohort_time_unit'].max())
            cohort_tu_monotic_range = self.cohort_period.generte_monotic_time_range(cohort_tu_min, cohort_tu_max) \
                .rename(columns={time_unit_name: 'cohort_time_unit'})

            pivot_template = pd.merge(
                cohort_group_monotic_range,
                cohort_tu_monotic_range,
                how='cross'
            )
            pivot_template = pivot_template[pivot_template['cohort_time_unit'] >= pivot_template['cohort_group']]

        else:
            cohort_per_min, cohort_per_max = cohorts_data['cohort_period'].min(), cohorts_data['cohort_period'].max()
            cohort_per_monotic_range = pd.Series(list(range(cohort_per_min, cohort_per_max + 1))).to_frame(
                name='cohort_period')
            pivot_template = pd.merge(
                cohort_group_monotic_range,
                cohort_per_monotic_range,
                how='cross'
            )

        return pivot_template

    def _expand_cohort_table(self, cohort_table: pd.DataFrame, represent_by: str,
                             represent_by_col: str, values_col: str) -> pd.DataFrame:
        """Expands the cohort table to include all possible time periods.

        Args:
            cohort_table: Raw cohort data.
            represent_by: How cohorts are represented ('time_unit' or 'period').
            represent_by_col: Name of the column representing cohorts.
            values_col: Name of the column containing values.

        Returns:
            Expanded cohort table.
        """
        pivot_template = self._prepare_pivot_template(cohort_table, represent_by)

        cohort_table = pd.merge(
            pivot_template,
            cohort_table,
            how='left',
            on=['cohort_group', represent_by_col]
        )
        cohort_table[values_col] = cohort_table[values_col].fillna(0)

        cohort_table = cohort_table.pivot_table(
            index='cohort_group',
            columns=represent_by_col,
            values=values_col,
            aggfunc=lambda x: x
        )

        return cohort_table

    def _check_fit_params(
            self,
            data: Union[pd.DataFrame, 'EventFrame'],
            represent_by: RepresentationTypes
    ):
        """Validates input parameters for fit methods.

        Args:
            data: Input data to validate.
            represent_by: Representation type to validate.

        Raises:
            ValueError: If parameters are invalid.
        """
        if not isinstance(data, EventFrame):
            raise ValueError('data must be an EventFrame')

        if not represent_by in get_args(self.RepresentationTypes):
            raise ValueError(f'Invalid representation type: {represent_by}')






