import pandas as pd
from typing import Dict, Optional, List, Union

from .cols_schema import EventFrameColsSchema
from .event_type import EventType


class EventFrame:

    def __init__(self, data: pd.DataFrame, cols_schema:  Union[Dict[str, str], 'EventFrameColsSchema'],
                 custom_cols: Optional[List] = None, prepare: bool = True):
        """Initializes the EventFrame class.

        Args:
            data (pd.DataFrame): Data to be stored in the EventFrame.
            cols_schema (Union[Dict[str, str], EventFrameColsSchema]): Schema for column names.
            custom_cols (Optional[List], optional): Additional custom column names. Defaults to None.
            prepare (bool, optional): If True, prepares the data. Defaults to True.

        Raises:
            ValueError: If the necessary fields are missing in cols_schema when prepare is False.
        """
        self.data = data.copy()
        self.cols_schema = EventFrameColsSchema(cols_schema)
        if prepare:
            self.prepare()
        else:
            if cols_schema.event_id is None or cols_schema.event_type is None or cols_schema.event_type_index is None:
                raise ValueError('Necessary fields are missing in cols_schema when prepare is False')

    def prepare(self):
        """Prepares the data by adding necessary columns."""
        if self.cols_schema.event_id is None:
            self._add_event_id()
        if self.cols_schema.event_type is None:
            self._add_event_type()
        if self.cols_schema.event_type_index is None:
            self._add_event_type_index()

        self.data[self.cols_schema.event_timestamp] = pd.to_datetime(self.data[self.cols_schema.event_timestamp])

    def to_dataframe(self) -> pd.DataFrame:
        """Gets the DataFrame.

        Returns:
            pd.DataFrame: The DataFrame containing the data.
        """
        sort_cols_list = [self.cols_schema.user_id, self.cols_schema.event_timestamp]
        if self.cols_schema.event_type_index is not None:
            sort_cols_list.append(self.cols_schema.event_type_index)
            
        return self.data.copy().sort_values(sort_cols_list)

    def filter(self, conditions: List[str], inplace: bool = False) -> 'EventFrame':
        """Filters the DataFrame rows based on specified conditions.

        Args:
            conditions (List[str]): A list of string expressions representing filter conditions.
            inplace (bool, optional): If True, modifies the current EventFrame. Defaults to False.

        Returns:
            EventFrame: A new EventFrame object with filtered data.
        """
        # Combine all conditions into single
        combined_condition = " & ".join(conditions)

        # Filter data
        filtered_data = self.data.query(combined_condition)

        if inplace:
            self.data = filtered_data

        # Return new filtered EventFrame
        return EventFrame(filtered_data, self.cols_schema)

    def add_col(self, col_name: str, col_data: Union[pd.Series, List]):
        """Adds a new column to the EventFrame.

        Args:
            col_name (str): The name of the new column.
            col_data (Union[pd.Series, List]): The data for the new column.

        Raises:
            ValueError: If the length of col_data does not match the number of rows in the data.
        """
        if len(col_data) != self.data.shape[0]:
            raise ValueError('The length of col_data does not match the number of rows in the data')

        self.data[col_name] = col_data

    def copy(self) -> 'EventFrame':
        """Creates a deep copy of the EventFrame object.

        Returns:
            EventFrame: A new EventFrame instance that is a copy of the current one.
        """
        copied_data = self.data.copy()
        copied_cols_schema = self.cols_schema.copy()

        return EventFrame(copied_data, copied_cols_schema)

    def _add_event_type(self) -> None:
        """Adds the event type column to the DataFrame."""
        data_len = self.data.shape[0]
        event_types = [EventType.RAW.value.name] * data_len
        event_type_col_name = 'event_type'

        self.add_col(event_type_col_name, event_types)
        self.cols_schema.event_type = event_type_col_name

    def _add_event_type_index(self) -> None:
        """Adds the event type index column to the DataFrame."""
        data_len = self.data.shape[0]
        event_type_idx = [EventType.RAW.value.order] * data_len
        event_type_index_col_name = 'event_type_index'

        self.add_col(event_type_index_col_name, event_type_idx)
        self.cols_schema.event_type_index = event_type_index_col_name

    def _add_event_id(self) -> None:
        """Adds the event ID column to the DataFrame."""
        # TODO: make id by hash of necessary columns for every new string
        data_len = self.data.shape[0]
        event_ids = list(range(data_len))
        event_id_col_name = 'event_id'

        self.add_col(event_id_col_name, event_ids)
        self.cols_schema.event_id = event_id_col_name

    def __repr__(self):
        """String representation of the EventFrame object.

        Returns:
            str: A string representation of the EventFrame.
        """
        return f"EventFrame(data={self.data.shape[0]} rows, columns={self.data.columns.tolist()})"

