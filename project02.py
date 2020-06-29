
import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------
    

def get_san(infp, outfp):
    """
    get_san takes in a filepath containing all flights 
    and an filepath where filtered dataset #1 is written (that is,
    All flights arriving or departing from San Diego Airport in 2015).
    The function should return None

    :Example:
    >>> infp = os.path.join('data', 'flights.test')
    >>> outfp = os.path.join('data', 'santest.tmp')
    >>> get_san(infp, outfp)
    >>> df = pd.read_csv(outfp)
    >>> df.shape
    (53, 31)
    >>> os.remove(outfp)
    """
    L = pd.read_csv(infp, chunksize = 10000)
    
    for df in L:
        #get locations of airports that contain SAN
        from_san = df['ORIGIN_AIRPORT'] == 'SAN'
        to_san = df['DESTINATION_AIRPORT'] == 'SAN'
        san = np.logical_or(from_san, to_san)

        san_df = df.loc[san]
        
        #append to csv for chunk
        with open(outfp, 'a') as f:
            san_df.to_csv(f, index=False, header = f.tell() == 0)
        
    return None

def get_sw_jb(infp, outfp):
    """
    get_san takes in a filepath containing all flights 
    and an filepath where filtered dataset #2 is written (that is,
    All flights flown by either JetBlue or Southwest Airline in 2015).
    The function should return None

    :Example:
    >>> infp = os.path.join('data', 'flights.test')
    >>> outfp = os.path.join('data', 'jbswtest.tmp')
    >>> get_sw_jb(infp, outfp)
    >>> df = pd.read_csv(outfp)
    >>> df.shape
    (73, 31)
    >>> os.remove(outfp)
    """
    L = pd.read_csv(infp, chunksize = 10000)

    for df in L:
        #get locations of JetBlue or SouthWest Airline
        jb = df['AIRLINE'] == 'B6'
        sw = df['AIRLINE'] == 'WN'
        jb_sw = np.logical_or(jb, sw)

        jb_sw_df = df.loc[jb_sw]
        
        #append to csv for chunk
        with open(outfp, 'a') as f:
            jb_sw_df.to_csv(f, index=False, header = f.tell() == 0)

    return None

# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------


def data_kinds():
    """
    data_kinds outputs a (hard-coded) dictionary of data kinds, 
    keyed by column name, 
    with values Q, O, N (for 'Quantitative', 'Ordinal', or 'Nominal').

    :Example:
    >>> out = data_kinds()
    >>> isinstance(out, dict)
    True
    >>> set(out.values()) == {'O', 'N', 'Q'}
    True
    """
    kind_dict = {'YEAR': 'O', 
                 'MONTH': 'O', 
                 'DAY': 'O', 
                 'DAY_OF_WEEK': 'O', 
                 'AIRLINE': 'O', 
                 'FLIGHT_NUMBER': 'N',
                 'TAIL_NUMBER': 'N', 
                 'ORIGIN_AIRPORT': 'N', 
                 'DESTINATION_AIRPORT': 'N',
                 'SCHEDULED_DEPARTURE': 'O',
                 'DEPARTURE_TIME': 'Q', 
                 'DEPARTURE_DELAY': 'Q', 
                 'TAXI_OUT': 'O',
                 'WHEELS_OFF': 'O',
                 'SCHEDULED_TIME': 'O',
                 'ELAPSED_TIME': 'Q', 
                 'AIR_TIME': 'Q',
                 'DISTANCE': 'Q', 
                 'WHEELS_ON': 'O',
                 'TAXI_IN': 'O',
                 'SCHEDULED_ARRIVAL': 'O',
                 'ARRIVAL_TIME': 'Q',
                 'ARRIVAL_DELAY': 'Q', 
                 'DIVERTED': 'N', 
                 'CANCELLED': 'N', 
                 'CANCELLATION_REASON': 'N',
                 'AIR_SYSTEM_DELAY': 'Q', 
                 'SECURITY_DELAY': 'Q', 
                 'AIRLINE_DELAY': 'Q',
                 'LATE_AIRCRAFT_DELAY': 'Q',
                 'WEATHER_DELAY': 'Q'
                }
    return kind_dict


def data_types():
    """
    data_types outputs a (hard-coded) dictionary of data types, 
    keyed by column name, with values str, int, float.

    :Example:
    >>> out = data_types()
    >>> isinstance(out, dict)
    True
    >>> set(out.values()) == {'int', 'str', 'float', 'bool'}
    True
    """

    type_dict = {'YEAR': 'int', 
                 'MONTH': 'int', 
                 'DAY': 'int', 
                 'DAY_OF_WEEK': 'int', 
                 'AIRLINE': 'str', 
                 'FLIGHT_NUMBER': 'int',
                 'TAIL_NUMBER': 'str', 
                 'ORIGIN_AIRPORT': 'str', 
                 'DESTINATION_AIRPORT': 'str',
                 'SCHEDULED_DEPARTURE': 'int', 
                 'DEPARTURE_TIME': 'float', 
                 'DEPARTURE_DELAY': 'float', 
                 'TAXI_OUT': 'float',
                 'WHEELS_OFF': 'float', 
                 'SCHEDULED_TIME': 'int',
                 'ELAPSED_TIME': 'float', 
                 'AIR_TIME': 'float',
                 'DISTANCE': 'int', 
                 'WHEELS_ON': 'float', 
                 'TAXI_IN': 'float',
                 'SCHEDULED_ARRIVAL': 'int', 
                 'ARRIVAL_TIME': 'float',
                 'ARRIVAL_DELAY': 'float', 
                 'DIVERTED': 'bool', 
                 'CANCELLED': 'bool', 
                 'CANCELLATION_REASON': 'str',
                 'AIR_SYSTEM_DELAY': 'float', 
                 'SECURITY_DELAY': 'float', 
                 'AIRLINE_DELAY': 'float',
                 'LATE_AIRCRAFT_DELAY': 'float',
                 'WEATHER_DELAY': 'float'
                }
    return type_dict

# ---------------------------------------------------------------------
# Question #3
# ---------------------------------------------------------------------


def basic_stats(flights):
    """
    basic_stats that takes flights and outputs a dataframe 
    that contains statistics for flights arriving/departing for SAN. 
    That is, the output should have two columns, ARRIVING and DEPARTING, 
    and be indexed as follows:

    * number of arriving/departing flights to/from SAN (count).
    * average flight (arrival) delay of arriving/departing 
    flights to/from SAN (mean).
    * median flight (arrival) delay of arriving/departing flights 
    to/from SAN (median).
    * the airline code of the airline that most often arrives/departs 
    to/from SAN (code).
    * the proportion of arriving/departing flights to/from SAN that 
    are canceled (canceled).
    * the airline code of the airline with the longest flight 
    (arrival) delay among all flights arriving/departing to/from SAN (max).

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> dtypes = data_types()
    >>> flights = pd.read_csv(fp, dtype=dtypes)
    >>> out = basic_stats(flights)
    >>> out.index.tolist() == ['ARRIVING', 'DEPARTING']
    True
    >>> cols = ['count', 'mean', 'median', 'code', 'canceled', 'max']
    >>> out.columns.tolist() == cols
    True
    """
    #separate arriving and departing
    arriving_df = flights.loc[flights['DESTINATION_AIRPORT'] == 'SAN']
    departing_df = flights.loc[flights['ORIGIN_AIRPORT'] == 'SAN']

    #index 0 for arriving, index 1 for departing
    count = [len(arriving_df),
             len(departing_df)]
    mean = [arriving_df['ARRIVAL_DELAY'].mean(),
            departing_df['ARRIVAL_DELAY'].mean()]
    median = [arriving_df['ARRIVAL_DELAY'].median(),
              departing_df['ARRIVAL_DELAY'].median()]
    code = [arriving_df['AIRLINE'].value_counts().keys()[0],
            departing_df['AIRLINE'].value_counts().keys()[0]]
    canceled = [arriving_df['CANCELLED'].sum() / len(arriving_df),
                departing_df['CANCELLED'].sum() / len(departing_df)]
    mx = [arriving_df.loc[arriving_df['ARRIVAL_DELAY'] == arriving_df['ARRIVAL_DELAY'].max(), 'AIRLINE'].iloc[0],
           departing_df.loc[departing_df['ARRIVAL_DELAY'] == departing_df['ARRIVAL_DELAY'].max(), 'AIRLINE'].iloc[0]]

    data = {'count': count,
            'mean': mean,
            'median': median,
            'code': code,
            'canceled': canceled,
            'max': mx}

    df = pd.DataFrame(data=data, index = ['ARRIVING', 'DEPARTING'])
    return df

# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------


def depart_arrive_stats(flights):
    """
    depart_arrive_stats that takes in a dataframe like 
    flights and calculates the following quantities 
    in a series (with index in parentheses):
    - The proportion of flights from/to SAN that 
      leave late, but arrive early or on-time (late1).
    - The proportion of flights from/to SAN that 
      leaves early, or on-time, but arrives late (late2).
    - The proportion of flights from/to SAN that 
      both left late and arrived late (late3).

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> dtypes = data_types()
    >>> flights = pd.read_csv(fp, dtype=dtypes)
    >>> out = depart_arrive_stats(flights)
    >>> out.index.tolist() == ['late1', 'late2', 'late3']
    True
    >>> isinstance(out, pd.Series)
    True
    >>> out.max() < 0.30
    True
    """

    #there is departure delay, but no arrival delay
    late1 = np.logical_and(flights['DEPARTURE_DELAY'] > 0, flights['ARRIVAL_DELAY'] <= 0).sum() / len(flights)
    #there is no departure delay, but there is arrival delay
    late2 = np.logical_and(flights['DEPARTURE_DELAY'] <= 0, flights['ARRIVAL_DELAY'] > 0).sum() / len(flights)
    #there is departure delay and an arrival delay
    late3 = np.logical_and(flights['DEPARTURE_DELAY'] > 0, flights['ARRIVAL_DELAY'] > 0).sum() / len(flights)
    return pd.Series([late1, late2, late3], index=['late1', 'late2', 'late3'])


def depart_arrive_stats_by_airline(flights):
    """

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> dtypes = data_types()
    >>> flights = pd.read_csv(fp, dtype=dtypes)
    >>> out = depart_arrive_stats_by_airline(flights)
    >>> out.columns.tolist() == ['late1', 'late2', 'late3']
    True
    >>> out.shape[0]
    12
    """
    
    #groupby airline and get departure stats
    airline_group = flights.groupby('AIRLINE')
    return airline_group.apply(depart_arrive_stats)


# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------


def cnts_by_airline_dow(flights):
    """
    cnts_by_airline_dow that takes in a dataframe 
    like flights and outputs a dataframe with
    - a column for each distinct value of AIRLINE,
    - a row for each day of the week, and
    - entries that give the total number of flights 
    that airline has on that day of the week over 2015.
    :param flights: a dataframe similar to flights.
    :returns: a dataframe of counts as above.

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = cnts_by_airline_dow(flights)
    >>> set(out.columns) == set(flights['AIRLINE'].unique())
    True
    >>> set(out.index) == set(flights['DAY_OF_WEEK'].unique())
    True
    >>> (out >= 0).all().all()
    True
    """

    #use airline as coulumns, day of week as index, and get counts
    return pd.pivot_table(flights, values = 'YEAR', columns='AIRLINE', index='DAY_OF_WEEK', aggfunc= 'count')


def mean_by_airline_dow(flights):
    """

    mean_by_airline_dow that takes in a dataframe 
    like flights and outputs a dataframe with
    - a column for each distinct value of AIRLINE,
    - a row for each day of the week, and
    - entries that give the average ARRIVAL_DELAY for 
    the flights of each airline on that day of the week.

    :param flights: a dataframe similar to `flights`.
    :returns: a dataframe of means as above.

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = mean_by_airline_dow(flights)
    >>> set(out.columns) == set(flights['AIRLINE'].unique())
    True
    >>> set(out.index) == set(flights['DAY_OF_WEEK'].unique())
    True
    """

    #get mean arrival delay by airline for each day of the week
    return pd.pivot_table(flights, values = 'ARRIVAL_DELAY', columns='AIRLINE', index='DAY_OF_WEEK', aggfunc= np.mean)


# ---------------------------------------------------------------------
# Question #6
# ---------------------------------------------------------------------

def predict_null_arrival_delay(row):
    """
    predict_null takes in a row of the flights data 
    (that is, a Series) and returns True if the 
    ARRIVAL_DELAY is null and otherwise False. Since the 
    function doesn't depend on ARRIVAL_DELAY, it should 
    work a row even if that index is dropped.

    :param row: a Series that represents a row of `flights`
    :returns: a boolean representing when `ARRIVAL_DELAY` is null.

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = flights.drop('ARRIVAL_DELAY', axis=1).apply(predict_null_arrival_delay, axis=1)
    >>> set(out.unique()) - set([True, False]) == set()
    True
    """

    #arrival delay missing by design if flight is diverted or cancelled
    return row['DIVERTED'] == True or row['CANCELLED'] == True


def predict_null_airline_delay(row):
    """
    predict_null takes in a row of the flights data
    (that is, a Series) and returns True if the
    AIRLINE_DELAY is null and otherwise False. Since the
    function doesn't depend on AIRLINE_DELAY, it should
    work a row even if that index is dropped.

    :param row: a Series that represents a row of `flights`
    :returns: a boolean representing when `AIRLINE_DELAY` is null.

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = flights.drop('AIRLINE_DELAY', axis=1).apply(predict_null_airline_delay, axis=1)
    >>> set(out.unique()) - set([True, False]) == set()
    True
    """

    #airline delay is missing by design if arrival delay is null or greater than 15
    return predict_null_arrival_delay(row) or row['ARRIVAL_DELAY'] < 15


# ---------------------------------------------------------------------
# Question #7
# ---------------------------------------------------------------------


def perm4missing(flights, col, N):
    """
    perm4missing takes in flights, 
    a column col, 
    and a number N and
    returns the p-value of the test 
    (using N simulations) that determines 
    if DEPARTURE_DELAY is MAR dependent on col.

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = perm4missing(flights, 'AIRLINE', 100)
    >>> 0 <= out <= 1
    True
    """
    
    """
    helper function get_tvd takes in a dataframe and column and returns
    the total variation distance
    
    :param df: a DataFrame of flights
    :param col: variation from column
    :returns: the total variation distance of column in dataframe
    """
    def get_tvd(df, col):
        pivot = (
        df
        .pivot_table(index='is_null', columns=col, aggfunc='size')
        .apply(lambda x:x / x.sum(), axis=1)
        )
        return pivot.diff().iloc[-1].abs().sum() / 2
    """
    helper function sim_null takes in a dataframe and column and returns the test statistic TVD of
    one instance of the null hypothesis
    
    :param df: a DataFrame of flights
    :param col: column of interest
    :returns: total variation distance of column in dataframe under null hypothesis
    """
    def sim_null(flights, col):
        shuffled_col = (
            flights[col]
            .sample(replace=False, frac=1)
            .reset_index(drop=True)
        )

        shuffled = (
            flights
            .assign(**{
                col: shuffled_col, 
                'is_null': flights['DEPARTURE_DELAY'].isnull()
            })
        )

        return get_tvd(shuffled, col)
    
    #get observed statistic
    obs_flights = flights.assign(is_null = flights['DEPARTURE_DELAY'].isnull())
    observed_tvd = get_tvd(obs_flights, col)
    
    #simulate null N times
    tvds = []
    for _ in range(N):
        tvd = sim_null(flights, col)
        tvds.append(tvd)
    
    #get pvalue
    pval = np.mean(tvds > observed_tvd)
    
    return pval


def dependent_cols():
    """
    Gives a list of columns on which DEPARTURE_DELAY 
    is MAR and dependent.

    :Example:
    >>> out = dependent_cols()
    >>> isinstance(out, list)
    True
    >>> cols = 'YEAR DAY_OF_WEEK AIRLINE DIVERTED CANCELLATION_REASON'.split()
    >>> set(out) <= set(cols)
    True
    """
    
    return ['DIVERTED', 'CANCELLATION_REASON']


def missing_types():
    """

    missing_types returns a Series, 
    - indexed by the following columns of flights: 
    CANCELLED, CANCELLATION_REASON, TAIL_NUMBER, ARRIVAL_TIME.
    - The values contain the most-likely missingness type of each column.
    - The unique values of this Series should be MD, MCAR, MAR, MNAR, NaN.

    :param:
    :returns: A series with index and values as described above.

    :Example:
    >>> out = missing_types()
    >>> isinstance(out, pd.Series)
    True
    >>> set(out.unique()) - set(['MD', 'MCAR', 'MAR', 'MNAR', np.NaN]) == set()
    True
    """
    ans = pd.Series(
            {'CANCELLED': np.NaN,
             'CANCELLATION_REASON': 'MD',
             'TAIL_NUMBER': 'MCAR',
             'ARRIVAL_TIME': 'MAR'
            })
    return ans


# ---------------------------------------------------------------------
# Question #8
# ---------------------------------------------------------------------

def prop_delayed_by_airline(jb_sw):
    """

    prop_delayed_by_airline that takes in a dataframe 
    like jb_sw and returns a DataFrame indexed by airline 
    that contains the proportion of each airline's flights 
    that are delayed.

    :param jb_sw: a dataframe similar to jb_sw
    :returns: a dataframe as above

    :Example:
    >>> fp = os.path.join('data', 'jetblue_or_sw.csv')
    >>> jb_sw = pd.read_csv(fp, nrows=100)
    >>> out = prop_delayed_by_airline(jb_sw)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> (out >= 0).all().all() and (out <= 1).all().all()
    True
    >>> len(out.columns) == 1
    True
    """
    #get proportion of flights that are delayed for each airline
    temp = jb_sw.fillna(0)
    return temp.groupby('AIRLINE').agg({'DEPARTURE_DELAY': lambda x: (x > 0).sum() / len(x)})


def prop_delayed_by_airline_airport(jb_sw):
    """
    prop_delayed_by_airline_airport that takes in a 
    dataframe like jb_sw and returns a DataFrame, with 
    columns given by airports, indexed by airline, that 
    contains the proportion of each airline's flights 
    that are delayed at each airport.

    :param jb_sw: a dataframe similar to jb_sw
    :returns: a dataframe as above.

    :Example:
    >>> fp = os.path.join('data', 'jetblue_or_sw.csv')
    >>> jb_sw = pd.read_csv(fp, nrows=100)
    >>> out = prop_delayed_by_airline_airport(jb_sw)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> ((out >= 0) | (out <= 1) | (out.isnull())).all().all()
    True
    >>> len(out.columns) == 6
    True
    """

    """
    helper function determines if the row originates from selected airports
    
    :param row: flight
    :returns: bool value, True if the flight originated from select airport, False otherwise
    """
    def origin_airport(row):
        airports = ['ABQ', 'BDL', 'BUR', 'DCA', 'MSY', 'PBI', 'PHX', 'RNO', 'SJC', 'SLC']
        return row['ORIGIN_AIRPORT'] in airports
    
    #get dataframe with only selected airports
    airport_df = jb_sw.loc[jb_sw.apply(origin_airport, axis = 1)]
    airport_df = airport_df.fillna(0)
    
    #get proportion of flights that are delayed for each airline by airport
    df = (
        airport_df.groupby(['AIRLINE', 'ORIGIN_AIRPORT'])
        .agg({'DEPARTURE_DELAY': lambda x: (x > 0).sum() / len(x)})
        .pivot_table(index='AIRLINE', columns='ORIGIN_AIRPORT', values='DEPARTURE_DELAY')
    )
    
    return df


# ---------------------------------------------------------------------
# Question #9
# ---------------------------------------------------------------------


def verify_simpson(df, group1, group2, occur):
    """
    Verifies that a dataset displays Simpson's Paradox.

    :param df: a dataframe
    :param group1: the first group being aggregated
    :param group2: the second group being aggregated
    :param occur: a column of df with values {0,1}, denoting
    if an event occurred.
    :returns: a boolean. True if simpson's paradox is present,
    otherwise False.

    :Example:
    >>> df = pd.DataFrame([[4,2,1], [1,2,0], [1,4,0], [4,4,1]], columns=[0,1,2])
    >>> verify_simpson(df, 0, 1, 2) in [True, False]
    True
    >>> verify_simpson(df, 0, 1, 2)
    False
    """
    #get proportion that occurance happens for group1
    first = df.groupby(group1).agg({occur: lambda x: (x > 0).sum() / len(x)})
    #get the min and max indices to observe reversal
    lower_index = first[occur].idxmin()
    higher_index = first[occur].idxmax()
    
    #get proportion that occurance happend for group1 by group2
    second = (df.groupby([group1, group2])
      .agg({occur: lambda x: (x > 0).sum() / len(x)})
      .pivot_table(index=group1, columns=group2, values=occur)
     )
    
        #return if there is a reversal for every value of group2
    return (second.loc[higher_index] < second.loc[lower_index]).all()


# ---------------------------------------------------------------------
# Question #10 (EXTRA CREDIT)
# ---------------------------------------------------------------------

def search_simpsons(jb_sw, N):
    """
    search_simpsons takes in the jb_sw dataset
    and a number N, and returns a list of 
    N airports for which the proportion of 
    flight delays between JetBlue and Southwest 
    satisfies Simpson's Paradox.

    Only consider airports that have '3 letter codes',
    Only consider airports that have at least one JetBlue and Southwest flight.

    :Example:
    >>> fp = os.path.join('data', 'jetblue_or_sw.csv')
    >>> jb_sw = pd.read_csv(fp, nrows=1000)
    >>> pair = search_simpsons(jb_sw, 2)
    >>> len(pair) == 2
    True
    """

    return ...


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_san', 'get_sw_jb'],
    'q02': ['data_kinds', 'data_types'],
    'q03': ['basic_stats'],
    'q04': ['depart_arrive_stats', 'depart_arrive_stats_by_airline'],
    'q05': ['cnts_by_airline_dow', 'mean_by_airline_dow'],
    'q06': ['predict_null_arrival_delay', 'predict_null_airline_delay'],
    'q07': ['perm4missing', 'dependent_cols', 'missing_types'],
    'q08': ['prop_delayed_by_airline', 'prop_delayed_by_airline_airport'],
    'q09': ['verify_simpson'],
    'q10': ['search_simpsons']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """
    
    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True
