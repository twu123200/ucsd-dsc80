
import os
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------

def normalize_hw(grades):
    """
    normalize_hw takes in a dataframe like grades
    and outputs a dataframe of normalized HW grades.
    The output should not take the late penalty into account.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> hw = normalize_hw(grades)
    >>> np.all(hw.columns == ['hw0%d' % d for d in range(1,10)])
    True
    >>> len(grades) == len(hw)
    True
    >>> np.all(np.isclose(hw.mean(), 0.80, atol=10))
    True
    """
    hw_cols = []
    max_points = []

    for c in grades.columns:
        if(c in ['hw%0*d' % (2, d) for d in range(1,100)] ):
            hw_cols.append(c)
        if(c in ['hw%0*d - Max Points' % (2, d) for d in range(1,100)] ):
            max_points.append(c)

    
    hw = pd.DataFrame(columns = hw_cols)
    for i in range(len(hw_cols)):
        hw_col = hw_cols[i]
        mp_col = max_points[i]
        hw[hw_col] = grades[hw_col] / grades[mp_col]
        
    return hw


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------


def last_minute_submissions(grades):
    """
    last_minute_submissions takes in the dataframe 
    grades and a Series indexed by HW assignment that 
    contains the number of submissions that were turned 
    in on time by the student, yet marked 'late' by Gradescope.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = last_minute_submissions(grades)
    >>> isinstance(out, pd.Series)
    True
    >>> np.all(out.index == ['hw0%d' % d for d in range(1,10)])
    True
    >>> (out > 0).sum()
    8
    """
    hw_cols = []
    lateness_cols = []

    for c in grades.columns:
        if(c in ['hw%0*d' % (2, d) for d in range(1,100)] ):
            hw_cols.append(c)
        if(c in ['hw%0*d - Lateness (H:M:S)' % (2, d) for d in range(1,100)] ):
            lateness_cols.append(c)
    
    late_count = []
    for i in range(len(lateness_cols)):
        lateness_col = lateness_cols[i]

        late_time_split = grades.loc[grades[lateness_col] != '00:00:00', lateness_col].str.split(':')
        count = late_time_split.apply(lambda x: int(x[0]) < 9).sum()
        late_count.append(count)

    late = pd.Series(late_count, index = hw_cols)
    return late

# ---------------------------------------------------------------------
# Question #3
# ---------------------------------------------------------------------


def adjust_lateness(grades):
    """
    adjust_lateness takes in the dataframe like `grades` 
    and returns a dataframe of HW grades adjusted for 
    lateness according to the syllabus.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = adjust_lateness(grades)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> out.loc[20, 'hw01'] != (grades.loc[20, 'hw01'] * 100)
    True
    """

    hw_cols = []
    lateness_cols = []

    for c in grades.columns:
        if(c in ['hw%0*d' % (2, d) for d in range(1,100)] ):
            hw_cols.append(c)
        if(c in ['hw%0*d - Lateness (H:M:S)' % (2, d) for d in range(1,100)] ):
            lateness_cols.append(c)
    
    one_week = 168
    two_week = 336

    hw = normalize_hw(grades)
    
    def adjust(hms):
        index = next(index_iter)
        if(int(hms[0]) < one_week):
            hw.loc[index, col] = hw.loc[index, col] * 0.9
        elif(int(hms[0]) < two_week):
            hw.loc[index, col] = hw.loc[index, col] * 0.8
        else:
            hw.loc[index, col] = hw.loc[index, col] * 0.5

    for i in range(len(lateness_cols)):
        lateness_col = lateness_cols[i]

        marked_late_df = grades.loc[grades[lateness_col] != '00:00:00']
        late_time_split = marked_late_df[lateness_col].str.split(':')
        actually_late = late_time_split.apply(lambda x: int(x[0]) >= 9)
        actually_late_df = marked_late_df.loc[actually_late == True]

        col = hw_cols[i]
        index_iter = iter(actually_late_df[lateness_col].str.split(":").index)

        actually_late_df[lateness_col].str.split(":").apply(adjust)
    return hw

# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------


def hw_total(adjusted):
    """
    hw_total takes in a dataframe of lateness-adjusted 
    HW grades, and computes the total HW grade for 
    each student according to the syllabus.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> adj = adjust_lateness(grades)
    >>> out = hw_total(adj)
    >>> isinstance(out, pd.Series)
    True
    >>> np.all((0 <= out) & (1 >= out))
    True
    >>> out.notnull().all()
    True
    """
    
    adj = adjusted.fillna(0)
    
    def adjusted_mean(data):
        return (np.mean(data) * len(data) - data.min()) / (len(data) - 1)
    
    return adj.apply(adjusted_mean , axis=1)


# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------


def average_student(grades):
    """
    average_student takes in a dataframe 
    like `grades` and outputs the HW grade 
    of a student who received the average 
    grade on each HW assignment.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = average_student(grades)
    >>> import numbers
    >>> isinstance(out, numbers.Real)
    True
    >>> np.isclose(out, 80, atol=5)
    True
    """
    hw_cols = []

    for c in grades.columns:
        if(c in ['hw%0*d' % (2, d) for d in range(1,100)] ):
            hw_cols.append(c)
            
    total = []
    for i in range(len(hw_cols)):
        col = hw_cols[i]

        temp = grades.dropna()
        total.append(temp[col].mean())

    total.remove(min(total))
    return np.mean(total)

def higher_or_lower():
    """
    higher_or_lower returns either 'higher' or
    'lower' depending on whether a hypothetical
    average student does better on average than
    the average total HW score in the course.

    :Example:
    >>> higher_or_lower() in ['higher', 'lower']
    True
    """
    
    return 'higher'


# ---------------------------------------------------------------------
# Question #6
# ---------------------------------------------------------------------


def extra_credit_total(grades):
    """
    extra_credit_total takes in a dataframe like `grades` 
    and returns the total extra-credit grade as a 
    proportion between 0 and 1.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = extra_credit_total(grades)
    >>> isinstance(out, pd.Series)
    True
    >>> np.all((0 <= out) & (1 >= out))
    True
    """
    ec_cols = []
    for c in grades.columns:
        if("extra-credit" in c):
            ec_cols.append(c)
        if("checkpoint" in c):
            ec_cols.append(c)
    ec = []
    ec_max_points = []
    ec_lateness = []
    for c in ec_cols:
        if("Max Points" in c):
            ec_max_points.append(c)
        elif("Lateness" in c):
            ec_lateness.append(c)
        else:
            ec.append(c)
    return grades[ec].sum(axis=1) / grades[ec_max_points].sum(axis=1)


# ---------------------------------------------------------------------
# Question #7
# ---------------------------------------------------------------------

def normalize_proj(grades):
    proj_cols = []
    for c in grades.columns:
        if("project" in c):
            if("checkpoint" not in c):
                proj_cols.append(c)
    proj_points_cols = []
    proj_max_points_cols = []
    proj_lateness_cols = []
    for c in proj_cols:
        if('Max Points' in c):
            proj_max_points_cols.append(c)
        elif('Lateness' in c):
            proj_lateness_cols.append(c)
        else:
            proj_points_cols.append(c)

    proj_df = pd.DataFrame()
    for i in np.arange(1, len(proj_points_cols) + 1):

        proj_num = 'project{:02d}'.format(i)
        proj_points= []
        proj_max_points = []
        proj_lateness = []

        for j in np.arange(len(proj_points_cols)):
            if(proj_num in proj_points_cols[j]):
                proj_points.append(proj_points_cols[j])
            if(proj_num in proj_max_points_cols[j]):
                proj_max_points.append(proj_max_points_cols[j])
            if(proj_num in proj_lateness_cols[j]):
                proj_lateness.append(proj_lateness_cols[j])

        if(len(proj_points) == 0):
            break

        proj_df[proj_num] = grades[proj_points].sum(axis=1) / grades[proj_max_points].sum(axis=1)
    return proj_df

def total_points(grades):
    """
    total_points takes in grades and returns 
    the final course grades according to the syllabus.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points(grades)
    >>> isinstance(out, pd.Series)
    True
    >>> np.isclose(out.max(), 0.95, atol=0.05)
    True
    """
    hw_grade = adjust_lateness(grades)
    num_hw = hw_grade.shape[1]
    hw_grade = hw_total(hw_grade) + (extra_credit_total(grades) / num_hw)
    hw_grade = hw_grade * 0.20
    
    midterm_cols = []
    final_cols = []
    for c in grades.columns:
        if("Midterm" in c):
            midterm_cols.append(c)
        if("Final" in c):
            final_cols.append(c)

    midterm_grade = (grades[midterm_cols[0]] / grades[midterm_cols[1]]) * 0.20
    final_grade = (grades[final_cols[0]] / grades[final_cols[1]]) * 0.30

    proj_df = normalize_proj(grades)
    proj_grade = proj_df.mean(axis=1) * 0.30

    return hw_grade + proj_grade + midterm_grade + final_grade


def final_grades(total):
    """
    final_grades takes in the final course grades 
    as above and returns a Series of letter grades 
    given by the standard cutoffs.

    :Example:
    >>> out = final_grades(pd.Series([0.92, 0.81, 0.41]))
    >>> np.all(out == ['A', 'B', 'F'])
    True
    """
    def letter(final_grade):
        if(final_grade >= 0.9):
            return 'A'
        elif(final_grade >= 0.8):
            return 'B'
        elif(final_grade >= 0.7):
            return 'C'
        elif(final_grade >= 0.6):
            return 'D'
        else:
            return 'F'
    return total.apply(letter)


def letter_proportions(grades):
    """
    letter_proportions takes in the dataframe grades 
    and outputs a Series that contains the proportion
    of the class that received each grade.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = letter_proportions(grades)
    >>> np.all(out.index == ['B', 'C', 'A', 'D', 'F'])
    True
    >>> out.sum() == 1.0
    True
    """
    total = total_points(grades)
    grade = final_grades(total)
    return grade.value_counts() / grade.shape[0]


# ---------------------------------------------------------------------
# Question # 8
# ---------------------------------------------------------------------


def simulate_pval(grades, N):
    """
    simulate_pval takes in the number of 
    simulations N and grades and returns 
    the likelihood that the grade of juniors 
    was no better on average than the class 
    as a whole (i.e. calculate the p-value).

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = simulate_pval(grades, 100)
    >>> 0 <= out <= 0.1
    True
    """
    sophomores = grades.loc[grades['Level'] == "SO"]
    soph_grades = total_points(sophomores)
    observed_stat = np.mean(soph_grades)
    class_grades = total_points(grades)

    averages = []
    
    for i in range(N):
        random_sample = class_grades.sample(sophomores.shape[0], replace = False)
        curr_avg = np.mean(random_sample)
        averages.append(curr_avg)

    averages = np.array(averages)
    return np.count_nonzero(averages >= observed_stat) / N


# ---------------------------------------------------------------------
# Question # 9
# ---------------------------------------------------------------------


def get_assignment_proportions(grades):
    """
    get_assignment_proportions takes in grades 
    and returns a dictionary keyed by assignment name
    with values given by the proportion of 
    the final grade that assignment makes up.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = get_assignment_proportions(grades)
    >>> 'project01_free_response' in out.keys()
    True
    >>> 'project03_checkpoint01' in out.keys()
    True
    >>> np.isclose(sum(out.values()), 1.0222, atol=0.01)
    True
    """
    hw_cols = []
    ec_cols = []
    proj_cols = []
    proj_max_points_cols = []
    for c in grades.columns:
        if(c in ['hw%0*d' % (2, d) for d in range(1,100)] ):
            hw_cols.append(c)
        if("checkpoint" in c or 'extra-credit' in c):
            if('Max Points' not in c and 'Lateness' not in c):
                ec_cols.append(c)

        if("project" in c):
            if('checkpoint' not in c):
                if('Max Points' not in c and 'Lateness' not in c):
                    proj_cols.append(c)
                if('Max Points' in c):
                    proj_max_points_cols.append(c)

    num_hw = len(hw_cols)
    hw_prop = 0.2 / num_hw
    num_ec = len(ec_cols)

    proportions = dict()
    for c in hw_cols:
        proportions[c] = hw_prop
    for c in ec_cols:
        proportions[c] = hw_prop / num_ec

    proj_combined_cols = normalize_proj(grades).columns
    num_proj = len(proj_combined_cols)
    proj_prop = 0.3 / num_proj

    for proj_num in proj_combined_cols:
        part_of_proj = []
        for c in proj_max_points_cols:
            if proj_num in c:
                part_of_proj.append(c)

        proj_num_total = grades[part_of_proj].mean().sum()
        for i in np.arange(len(proj_cols)):
            col = proj_cols[i]
            max_point_col = proj_max_points_cols[i]
            weight = grades[max_point_col].mean()
            proportions[col] = (weight / proj_num_total) * proj_prop

    proportions['Midterm'] = 0.2
    proportions['Final'] = 0.3
    return proportions


# ---------------------------------------------------------------------
# Question #10
# ---------------------------------------------------------------------


def curved_total_points(grades):
    """
    curved_total_points takes in grades and outputs 
    the curved total scores for each student.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = curved_total_points(grades)
    >>> isinstance(out, pd.Series)
    True
    >>> out.max() < 2
    True
    >>> out.min() > -10
    True
    """
    
    su_df = pd.DataFrame()

    proportions = get_assignment_proportions(grades)
    assignment_cols = proportions.keys()

    for c in assignment_cols:
        su_df[c] = (grades[c] - grades[c].mean()) / grades[c].std()

    su_df = su_df.fillna(0)

    def curve(student):
        su_grade = 0
        for c in su_df.columns:
            su_grade += proportions[c] * student[c]
        return su_grade
    return su_df.apply(curve, axis=1)

def curved_letter_grades(curved_grades, prop):
    """
    curved_letter_grades which takes in:
        - a Series of curved course grades (as above),
        - a Series of letter grade distributions 
        (e.g. the output of letter_proportions)

    and returns a Series containing the letter grade of 
    each student according to the curve (as described in
    the notebook).

    :Example:
    >>> prop = pd.Series([0.2]*5, index='A B C D F'.split())
    >>> curved_grades = pd.Series([-0.2, 0, -0.5, 0.2, 2, -1, -3.1, 3, 0.4, 5])
    >>> out = curved_letter_grades(curved_grades, prop)
    >>> isinstance(out, pd.Series)
    True
    >>> distr = out.value_counts(normalize=True).sort_index()
    >>> np.all(distr == prop.sort_index())
    True
    >>> out.iloc[1] == 'C'
    True
    """
    def curved_letter(curved_final_grade):
        if(curved_final_grade >= cutoff[0]):
            return 'A'
        elif(curved_final_grade >= cutoff[1]):
            return 'B'
        elif(curved_final_grade >= cutoff[2]):
            return 'C'
        elif(curved_final_grade >= cutoff[3]):
            return 'D'
        else:
            return 'F'

    prop = prop.sort_index()
    cutoff = []
    percentile = 100
    for prop in prop:
        percentile -= (prop * 100)
        cutoff.append(np.percentile(curved_grades, percentile))

    return curved_grades.apply(curved_letter)


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['normalize_hw'],
    'q02': ['last_minute_submissions'],
    'q03': ['adjust_lateness'],
    'q04': ['hw_total'],
    'q05': ['average_student', 'higher_or_lower'],
    'q06': ['extra_credit_total'],
    'q07': ['total_points', 'final_grades'],
    'q08': ['simulate_pval'],
    'q09': ['get_assignment_proportions'],
    'q10': ['curved_total_points', 'curved_letter_grades']
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
