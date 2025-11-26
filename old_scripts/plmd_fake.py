import re
import gzip
import sys
import pandas as pd

class Constants(list):
    """Custom class used to store plumed constants.
    """
    def __init__(self, l):
        if isinstance(l, dict):
            for k in l:
                self.append((k, l[k]))
        else:
            self.extend(l)
        for i, item in enumerate(self):
            if len(item) == 2:
                self[i] = (self[i][0], self[i][1], str(self[i][1]))
            elif len(item) ==3:
                pass
            else:
                raise ValueError("plumed.Constants should be initialized with a list of 2- or 3-plets")

class PlumedSeries(pd.Series):
    @property
    def _constructor(self):
        return PlumedSeries
    @property
    def _constructor_expanddim(self):
        return PlumedDataFrame
class PlumedDataFrame(pd.DataFrame):
    _metadata=["plumed_constants"]
    @property
    def _constructor(self):
        return PlumedDataFrame
    @property
    def _constructor_sliced(self):
        return PlumedSeries

def _fix_file(file, mode):
    """Internal utility: returns a file open with mode.
       Takes care of opening file (if it receives a string)
       and or unzipping (if the file has ".gz" suffix).
    """
    # allow passing a string
    if isinstance(file, str):
        file = open(file, mode)
    # takes care of gzipped files
    if re.match(".*\\.gz", file.name):
        file = gzip.open(file.name, mode)
    return file

def process_dataframe(df, enable_constants, constants):
    if enable_constants=='columns':
        for c in constants: df[c[0]] = c[1]
    if enable_constants=='metadata':
        df = PlumedDataFrame(df)
        df.plumed_constants = Constants(constants)
    return df

def read_as_pandas(file_or_path, enable_constants=True, usecols=None, skiprows=None, nrows=None, index_col=None):
    if enable_constants is True:
        enable_constants='metadata'
    if enable_constants is False:
        enable_constants='no'

    if not (enable_constants=='no' or enable_constants=='metadata' or enable_constants=='columns'):
        raise ValueError("enable_conversion not valid")

    file_or_path = _fix_file(file_or_path,'rt')
    line = file_or_path.readline()
    columns = line.split()

    if len(columns)<2:
        raise ValueError("Error reading PLUMED file "+file_or_path.name + ". Not enough columns")
    if columns[0] != "#!" or columns[1] != "FIELDS":
        raise ValueError("Error reading PLUMED file" +file_or_path.name + ". Columns: "+columns[0]+" "+columns[1])

    # read column names
    columns = columns[2:]

    # read constants
    constants=[]
    if enable_constants != 'no':
        while True:
            pos = file_or_path.tell()
            line = file_or_path.readline()
            file_or_path.seek(pos)
            if not line:
                break
            sets = line.split()
            if len(sets) < 4:
                break
            if sets[0]!="#!" or sets[1]!="SET":
                break
            v = sets[3]
            # name / value / string
            constants.append((sets[2], v, sets[3]))
            file_or_path.readline() # read again to go to next line

    # read the rest of the file
    # notice that if chunksize was provided the result will be an iterable object
    df = pd.read_csv(file_or_path, sep=r'\s+', comment="#", header=None, names=columns,
                    usecols=usecols, skiprows=skiprows, nrows=nrows, index_col=index_col)

    return process_dataframe(df, enable_constants, constants)

def write_pandas(df, file_or_path=None):
    # check if there is an index. if so, write it as an additional field
    has_index = hasattr(df.index, 'name') and df.index.name is not None
    # check if there is a multi-index
    has_mindex = (not has_index) and hasattr(df.index, 'names') and df.index.names[0] is not None
    # writing multiindex is currently not supported
    if has_mindex:
        raise TypeError("Writing dataframes with MultiIndexes is not supported at this time")
    # handle file
    if file_or_path is None:
        file_or_path = sys.stdout
    file_or_path = _fix_file(file_or_path, 'wt')
    # write header
    file_or_path.write("#! FIELDS")
    if has_index:
        file_or_path.write(" " + str(df.index.name))
    for n in df.columns:
        file_or_path.write(" " + str(n))
    file_or_path.write("\n")
    # write constants
    if hasattr(df, "plumed_constants") and isinstance(df.plumed_constants, Constants):
        for c in df.plumed_constants:
    # notice that string constants are written (e.g. pi) rather than the numeric ones (e.g. 3.14...)
            file_or_path.write("#! SET "+c[0]+" "+c[2]+"\n")
    # write data
    for i in range(df.shape[0]):
        if has_index:
            file_or_path.write(" " + str(df.index[i]))
        for j in df.columns:
            file_or_path.write(" " + str(df[j][i]))
        file_or_path.write("\n")
