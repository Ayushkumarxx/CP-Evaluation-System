import pandas as pd
from pathlib import Path
import traceback

# Enable future behavior for pandas downcasting safety
pd.set_option('future.no_silent_downcasting', True)


class DataReader:
    def __init__(self, file_path):
        """
        Initialize the DataReader with the file path.
        Loads data immediately and prepares test-specific slices.
        """
        self.file_path = Path(__file__).parent / "data" / file_path
        self.df = self.load_data()
        self.df_lists = self.slice_by_test_name()

    def load_data(self):
        """
        Load and clean the Excel file.
        Returns:
            pandas.DataFrame or None
        """
        try:
            df_raw = pd.read_excel(self.file_path, header=None)

            # Pre-clean all object columns
            object_cols = df_raw.select_dtypes(include='object').columns
            df_raw[object_cols] = df_raw[object_cols].apply(
                lambda col: col.astype(str).str.strip().str.upper()
            )

            # Identify header row using "SL.NO"
            header_row = df_raw[df_raw.eq('SL.NO').any(axis=1)].index[0]

            # Extract and clean header
            header = (
                df_raw.iloc[header_row]
                .astype(str)
                .str.replace('\n', ' ', regex=True)
                .str.strip()
                .str.upper()
            )

            # Extract content below header and assign new column names
            df = df_raw.iloc[header_row + 1:].copy()
            df.columns = header.tolist()
            df.reset_index(drop=True, inplace=True)

            return df

        except Exception as e:
            print(f"Error loading data: {e}")
            traceback.print_exc()
            return None  # Return None to indicate failure (used in other methods)

    def get_comman_cols(self, start=0, end=6):
        """
        Extract common columns based on fixed range.
        Used as left-side block for test slices.
        """
        return self.df.iloc[:, start:end] if self.df is not None else pd.DataFrame()

    def get_uncommon_cols(self, start=6):
        """
        Extract test-related columns.
        """
        return self.df.iloc[:, start:] if self.df is not None else pd.DataFrame()

    def slice_by_test_name(self, marker='TEST NAME', start=0, end=6):
        """
        Slices the dataset into blocks separated by repeated 'TEST NAME' columns.
        Returns:
            List of pandas.DataFrame blocks (does not return full data directly).

        DEV NOTE:
        - Each DataFrame block includes the fixed common fields + one test-specific block.
        - This approach helps modularize large datasets with multiple test columns.
        """
        if self.df is None:
            return []

        try:
            common_df = self.get_comman_cols(start=start, end=end)
            test_df = self.get_uncommon_cols(start=end)

            # Get positions of all TEST NAME markers
            marker_indices = [-1] + [
                i for i, col in enumerate(test_df.columns) if col == marker
            ]

            test_blocks = []

            for i in range(1, len(marker_indices)):
                slice_start = marker_indices[i - 1] + 1
                slice_end = marker_indices[i] + 1

                # Combine common + this test-specific block
                test_slice = test_df.iloc[:, slice_start:slice_end]
                full_block = pd.concat([common_df, test_slice], axis=1)

                test_blocks.append(full_block)

            return test_blocks  # DEV NOTE: List of blocks, not a single dataframe

        except Exception as e:
            print(f"Error while slicing test blocks: {e}")
            traceback.print_exc()
            return []
