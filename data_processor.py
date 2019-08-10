class Data_Processor:
    """Class to deal with any data processing"""
    encoders: dict = {}
    scalers: dict = {}

    @classmethod
    def contents(cls):
        """List all methods of class with their descriptions."""
        for method in dir(cls):
            if not method.startswith('__'):
                print(f'Method name: {method}\nMethod description: {getattr(cls, method).__doc__}\n\n')

    @classmethod
    def read_data(cls, file_name: str) -> Optional[DataFrame]:
        """Read in data from ``file_name`` and return a pandas DataFrame
        object filled with the data from the file.

        :param file_name: Name of file to read (should be csv for now)
        :type file_name: str
        :returns: Pandas DataFrame object if file_name is valid, else returns None.
        :rtype: DataFrame | None
        """
        try:
            df = read_csv(file_name)
            return df
        except Exception:
            print('An error has occurred. Perhaps the file name/path was incorrect?')
            return

    @classmethod
    def drop_columns(cls, data: DataFrame, columns_to_drop: List[str]) -> DataFrame:
        """Drop columns in ``columns_to_drop`` from ``data``

        :param data: Pandas DataFrame object.
        :type data: DataFrame
        :param columns_to_drop: List of column names (as strings) to drop from ``data``
        :type columns_to_drop: list
        :returns: Pandas DataFrame object without columns in ``columns_to_drop``.
        :rtype: DataFrame
        """
        reduced_df = data
        for col in columns_to_drop:
            try:
                reduced_df = reduced_df.drop(col, axis=1)
            except Exception:
                print(f'There was a problem removing {col} from the DataFrame.')

        return reduced_df

    @classmethod
    def drop_rows(cls, data: DataFrame, rows_to_drop: List[int]) -> DataFrame:
        """Drop rows in ``rows_to_drop`` from ``data``

        :param data: Pandas DataFrame object.
        :type data: DataFrame
        :param rows_to_drop: List of row indices to drop from ``data``
        :type rows_to_drop: list
        :returns: Pandas DataFrame object without rows indexed by values in ``rows_to_drop``.
        :rtype: DataFrame
        """
        dirty_rows = []
        for row in rows_to_drop:
            try:
                dirty_rows.append(data.index[row])
            except:
                print(f'There was a problem removing row {row} from the DataFrame.')

        return data.drop(dirty_rows)

    @classmethod
    def sanitize(cls, data: DataFrame) -> DataFrame:
        """Drop each row that contains a null value for a given attribute.

        :param data: Pandas DataFrame
        :type data: DataFrame
        :returns: Pandas DataFrame that has no null values.
        :rtype: DataFrame
        """
        return data.dropna(axis=0)

    @classmethod
    def scale_data(cls, data: DataFrame, scaler_map: Dict[str, Any]) -> DataFrame:
        """Scale columns in ``data`` according to scaler sc

        :param data: Pandas DataFrame
        :type data: DataFrame
        :param sc: Mapping from data column name to designated scaler class (DO NOT PUT
        AN INSTANCE OF YOUR CHOSEN CLASS IN ``scaler_map``)
        :type sc: dict
        :returns: Scaled DataFrame
        :rtype: DataFrame
        """
        cols = data.columns
        new_df = data
        for col in cols:
            if scaler_map.get(col):
                cls.scalers[col] = scaler_map[col]().fit(data[col])
                new_df = cls.scalers[col].transform(data[col])
        return new_df

    @classmethod
    def encode_categories(cls, data: DataFrame, threshold: int = 20, do_not_encode: Set[str] = set()) -> DataFrame:
        """Encode categorical attributes in ``data``.

        :param data: Pandas DataFrame
        :type data: DataFrame
        :param threshold: Determines how many unique category features should be encoded;
        this is based on how much memory your computer has.
        :type threshold: int
        :param do_not_encode: List of column names to not encode.
        :type do_not_encode: list
        :returns: DataFrame of encoded categories.
        :rtype: DataFrame
        """
        fields = set(data.columns.values) - do_not_encode
        print(fields)
        encoded_data = data
        for field in fields:
            if len(data[field].unique()) <= threshold:
                encoded_data[field] = get_dummies(data[field], drop_first=True)
            else:
                cls.encoders[field] = LabelEncoder().fit(data[field])
                encoded_data[field] = cls.encoders[field].transform(encoded_data[field])
        return encoded_data

    @classmethod
    def split(cls, data: DataFrame, split: float = 0.8) -> Tuple[DataFrame]:
        """Split ``data`` into testing/training sets.

        :param data: Pandas DataFrame to split
        :type data: DataFrame
        :param split: Decides what percentage of the elements in ``data`` will be in the training set.
        :type split:
        :returns:
        :rtype: tuple
        """
        if not 0 <= split <= 1.0:
            print('Incorrect value for ``split``. Expected a number in the range [0, 1.0]')
            return
        split_index = int(split*len(data))
        # TODO: FINISH THIS METHOD

    @classmethod
    def separate_io(cls, data: DataFrame, input_fields: List[str], output_fields: List[str]) -> Tuple[DataFrame]:
        """Separates ``data`` into input data and target data

        :param data: Pandas DataFrame
        :type data: DataFrame
        :param input_fields: List of input data field names. This is the data you want to input into your models.
        :type input_fields: list
        :param output_fields: List of target data field names. This is the data you want to predict.
        :type output_fields: list
        :returns: Binary tuple of Pandas DataFrames; the first element will be input data and the second will be
        target data.
        :rtype: tuple
        """
        pass
        # TODO: FINISH THIS METHOD
