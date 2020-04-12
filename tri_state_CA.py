import random
from matplotlib import pyplot as plt
# %matplotlib inline


def random_string(length):
    '''
    Returns a random bit string of the given length.

    Parameters
    ----------
    length: int
        Posivite integer that specifies the desired length of the bit string.

    Returns
    -------
    out: list
        The random bit string given as a list, with int elements.
    '''
    if not isinstance(length, int) or length < 0:
        raise ValueError("input length must be a positive ingeter")
    return [random.randint(0,2) for _ in range(length)]


def rule_in_base(rule_number, base, key_number):
    '''
    Convert the decimal integer rule number into the desired base value. Then
    zeros are padded to the left if required to euqate the total bit number to 
    key_number.

    Parameters
    ----------
    rule_number: int
        Positive decimal integer that specifies the rule number.

    base: int
        Positive integer the specifies the base or number of state of a cell.

    key_number: int
        Positive integer that specifies the total number of tuples in the 
        neighborhoods. We want the total bit number in the rule_number after 
        converting it to the desired base is equal to the tuples in the 
        neighborhoods or the total key number in the lookup table.

    Returns
    ----------
    rule_number_in_base : list
        The decimal rule_number converted to the desired base as each element 
        of the list represents each bit.
    '''
    if not isinstance(base, int) or base < 0:
        raise ValueError("base must be a positive int")

    rule_number_in_base = []
    dividend = rule_number
    quotient = rule_number
    while quotient >= base:
        quotient = dividend // base
        rule_number_in_base.insert(0, dividend % base)
        dividend = quotient
    rule_number_in_base.insert(0, quotient)
    #print(rule_number_in_base)

    # Added paranthesis in the following line so that it can be divided into
    # two lines without breaking continuation
    rule_number_in_base = ([0] * (key_number - len(rule_number_in_base)) 
                           + rule_number_in_base)
    #print(rule_number_in_base)

    return rule_number_in_base


def lookup_table(rule_number):
    '''
    Returns a dictionary which maps ECA neighborhoods to output values. This 
    is different than 2 state ECA case.

    Parameters
    ----------
    rule_number: int
        Integer value between 0 and 19682, inclusive.

    Returns
    -------
    lookup_table: dict
        Lookup table dictionary that maps neighborhood tuples to their output
        according to the ECA local evolution rule (i.e. the lookup table), as 
        specified by the rule number.
    '''
    if (not isinstance(rule_number, int) or rule_number < 0
            or rule_number > 19682):
        raise ValueError("rule_number must be an int between 0 and 19682,"
                         "inclusive")

    neighborhoods = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), 
                     (2,1), (2,2)]

    rule_number_in_base = rule_in_base(rule_number, base=3, 
                                       key_number=len(neighborhoods))


    return dict(zip(neighborhoods, rule_number_in_base))


def spacetime_diagram(spacetime_field, size=12, colors=plt.cm.Greys):
    '''
    Produces a simple spacetime diagram image using matplotlib imshow with 
    'nearest' interpolation.

    Parameters
    ---------
    spacetime_field: array-like (2D)
        1+1 dimensional spacetime field, given as a 2D array or list of lists.
        Time should be dimension 0; so that spacetime_field[t] is the spatial 
        configuration at time t.

    size: int, optional (default=12)
        Sets the size of the figure: figsize=(size,size)
    colors: matplotlib colormap, optional (default=plt.cm.Greys)
        See https://matplotlib.org/tutorials/colors/colormaps.html for 
        colormap choices. A colormap 'cmap' is called as: colors=plt.cm.cmap
    '''
    plt.figure(figsize=(size,size))
    plt.imshow(spacetime_field, cmap=colors, interpolation='nearest')
    #plt.show()
    plt.svaefig("tri_state_CA.pdf", dpi=100)


class TriStateCA(object):
    '''
    Elementary cellular automata simulator for tri state cell.
    '''
    def __init__(self, rule_number, initial_condition):
        '''
        Initializes the simulator for the given rule number and initial 
        condition.

        Parameters
        ----------
        rule_number: int
            Integer value between 0 and 19682, inclusive.
        initial_condition: list
             Elements of the list are ints.

        Attributes
        ----------
        lookup_table: dict
            Lookup table for the ECA given as a dictionary, with neighborhood 
            tuple keys.
        initial: array_like
            Copy of the initial conditions used to instantiate the simulator
        spacetime: array_like
            2D array (list of lists) of the spacetime field created by the 
            simulator.
        current_configuration: array_like
            List of the spatial configuration of the ECA at the current time
        '''
        for i in initial_condition:
            if i not in [0, 1, 2]:
                raise ValueError("initial condition must be a list of 0s, 1s"
                                 "and 2s")

        self.lookup_table = lookup_table(rule_number)
        self.initial = initial_condition
        self.spacetime = [initial_condition]
        self.current_configuration = initial_condition.copy()
        self._length = len(initial_condition)

    def evolve(self, time_steps):
        '''
        Evolves the current configuration of the ECA for the given number of 
        time steps.

        Parameters
        ----------
        time_steps: int
            Positive integer specifying the number of time steps for evolving
            the ECA.
        '''
        if time_steps < 0:
            raise ValueError("time_steps must be a non-negative integer")
        # try converting time_steps to int and raise a custom error if this 
        # can't be done
        try:
            time_steps = int(time_steps)
        except ValueError:
            raise ValueError("time_steps must be a non-negative integer")

        for _ in range(time_steps):#use underscore if the index willn't be used
            new_configuration = []
            for i in range(self._length):

                neighborhood = (self.current_configuration[(i-1)],
                                self.current_configuration[i])

                new_configuration.append(self.lookup_table[neighborhood])

            self.current_configuration = new_configuration
            self.spacetime.append(new_configuration)

