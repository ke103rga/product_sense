from typing import List, Dict, Union


class EventFrameColsSchema:
    """Class for storing schema of column names in an event frame.

    Attributes:
        _event_id (str): Name of the event ID column.
        _event_type (str): Name of the event type column.
        _event_type_index (str): Name of the event type index column.
        _event_name (str): Name of the event name column.
        _event_timestamp (str): Name of the event timestamp column.
        _user_id (str): Name of the user ID column.
        _session_id (str): Name of the session ID column.
        _cohort_group (str): Name of the cohort group column.
        _custom_cols (List[str]): List of custom column names.
    """
    _event_id: str
    _event_type: str
    _event_type_index: str
    _event_name: str
    _event_timestamp: str
    _user_id: str
    _session_id: str
    _cohort_group: str
    _custom_cols: List[str] = list()

    __necessary_cols = [
        'event_timestamp',
        'event_name',
        'user_id'
    ]

    def __init__(self, cols_schema: Union[Dict[str, str], 'EventFrameColsSchema']):
        """Initializes the EventFrameColsSchema with a given schema.

        Args:
            cols_schema (Union[Dict[str, str], EventFrameColsSchema]):
                Either a dictionary with column names or another EventFrameColsSchema.

        Raises:
            ValueError: If any necessary columns are missing from cols_schema.
        """
        if type(cols_schema) is EventFrameColsSchema:
            self._event_id = cols_schema._event_id
            self._event_type = cols_schema._event_type
            self._event_type_index = cols_schema._event_type_index
            self._event_name = cols_schema._event_name
            self._event_timestamp = cols_schema._event_timestamp
            self._user_id = cols_schema._user_id
            self._session_id = cols_schema._session_id
            self._cohort_group = cols_schema._cohort_group
            return

        if not set(self.__necessary_cols).issubset(set(cols_schema.keys())):
            missing_cols = set(self.__necessary_cols) - set(cols_schema.keys())
            raise ValueError(f"Missing necessary columns: {missing_cols}")

        self._event_id = cols_schema.get('event_id')
        self._event_type = cols_schema.get('event_type')
        self._event_type_index = cols_schema.get('event_type_index')
        self._event_name = cols_schema.get('event_name')
        self._event_timestamp = cols_schema.get('event_timestamp')
        self._user_id = cols_schema.get('user_id')
        self._session_id = cols_schema.get('session_id')
        self._cohort_group = cols_schema.get('cohort_group')

    def copy_from(self, other):
        for attr in dir(other):
            if not attr.startswith('__'):
                setattr(self, attr, getattr(other, attr))

    def check_setting_col_name(self, value):
        """Checks if the column name is valid.

        Args:
            value (str): The column name to validate.

        Raises:
            ValueError: If the name is not a string or is empty.
        """
        if not isinstance(value, str):
            raise ValueError("The  name should be string")
        if len(value) == 0:
            raise ValueError("The name shouldn't be empty")

    def copy(self) -> 'EventFrameColsSchema':
        """Creates a deep copy of the EventFrameColsSchema object.

        Returns:
            EventFrameColsSchema: A new instance of EventFrameColsSchema that is a copy of the current one.
        """

        copied_schema = {
            'event_id': self._event_id,
            'event_type': self._event_type,
            'event_type_index': self._event_type_index,
            'event_name': self._event_name,
            'event_timestamp': self._event_timestamp,
            'user_id': self._user_id,
            'session_id': self._session_id,
            'cohort_group': self._cohort_group,
            'custom_cols': self._custom_cols.copy()
        }

        return EventFrameColsSchema(copied_schema)

    @property
    def event_id(self) -> str:
        return self._event_id

    @event_id.setter
    def event_id(self, value: str):
        self.check_setting_col_name(value)
        self._event_id = value

    @property
    def event_type(self) -> str:
        return self._event_type

    @event_type.setter
    def event_type(self, value: str):
        self.check_setting_col_name(value)
        self._event_type = value

    @property
    def event_type_index(self) -> str:
        return self._event_type_index

    @event_type_index.setter
    def event_type_index(self, value: str):
        self.check_setting_col_name(value)
        self._event_type_index = value

    @property
    def event_name(self) -> str:
        return self._event_name

    @event_name.setter
    def event_name(self, value: str):
        self.check_setting_col_name(value)
        self._event_name = value

    @property
    def event_timestamp(self) -> str:
        return self._event_timestamp

    @event_timestamp.setter
    def event_timestamp(self, value: str):
        self.check_setting_col_name(value)
        self._event_timestamp = value

    @property
    def user_id(self) -> str:
        return self._user_id

    @user_id.setter
    def user_id(self, value: str):
        self.check_setting_col_name(value)
        self._user_id = value

    @property
    def session_id(self) -> str:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str):
        self.check_setting_col_name(value)
        self._session_id = value

    @property
    def cohort_group(self) -> str:
        return self._cohort_group

    @cohort_group.setter
    def cohort_group(self, value: str):
        self.check_setting_col_name(value)
        self._cohort_group = value

    @property
    def custom_cols(self) -> List[str]:
        return self._custom_cols

    @custom_cols.setter
    def custom_cols(self, value: List[str]):
        if not isinstance(value, list):
            raise ValueError("custom_cols должно быть списком.")
        self._custom_cols = value

    def __repr__(self):
        """
        String representation of EventFrameColsSchema.
        """
        return (f"EventFrameColsSchema(event_id={self.event_id}, "
                f"event_type={self.event_type}, "
                f"event_index={self.event_type_index}, "
                f"event_name={self.event_name}, "
                f"event_timestamp={self.event_timestamp}, "
                f"user_id={self.user_id}, "
                f"session_id={self.session_id}, "
                f"cohort_group={self.cohort_group}, "
                f"custom_cols={self.custom_cols})")






