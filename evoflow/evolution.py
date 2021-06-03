import tensorflow as tf
import numpy as np
import random

from deap import algorithms, base, creator, tools

from evoflow.network import MLP, MLPDescriptor, TCNN, CNN, RNN
from evoflow.mutation import MLP_Mutation, CNN_Mutation, TCNN_Mutation, RNN_Mutation
import os

from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.models import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

descs = {"ConvDescriptor": CNN, "MLPDescriptor": MLP, "TConvDescriptor": TCNN, "RNNDescriptor": RNN}


class DescriptorContainer(object):
    """
    Auxiliar class for DEAP algorithm. This object will represent the individual
    that will be evolved in the Genetic Algorithm. It contains a dictionary
    with the networks descriptors and the hyperparameters that will be evolved.
    This class does not require a fitness attribute because it will be added 
    later by the DEAP's creator.
    
    :param descriptor_list: A dictionary containing the descriptors of the 
                            networks and hyperparameters that will be evolved 
                            with DEAP algorithm.
    """
    def __init__(self, descriptor_list):
        self.descriptor_list = descriptor_list


class Evolving:
    """
    This is the main class in charge of evolving model descriptors. It contains 
    all the needed functions and atributes in order to realize the evolution.
    For evolving the desired network, these are the two stepts to follow:
    1.- Initialize this class with all the desired qualities.
    2.- Call evolve() function.
        
    :param n_inputs: List of lists with the dimensions of the input sizes of each network.
    :param n_outputs: List of lists with the dimensions of the output sizes of each network.
    :param max_num_layers: Maximum number of layers.
    :param max_num_neurons: Maximum number of neurons in all layers (only relevant with MLP descriptors).
    :param max_filter: Maximum size of filter (only relevant with CNN and TCNN descriptors).
    :param max_stride: Maximum size of stride (only relevant with CNN and TCNN descriptors).
    :param descriptors: List with all the network descriptors that are wanted to be evolved.
    :param evaluation: Function for evaluating the model. A string in simple cases ('MSE' or 'XEntropy')
                       and one defined by the user in complex cases.
    :param batch_size: Number of samples per batch are used during training process.
    :param lrate: Learning rate used during training process.
    :param iters: Number of iterations that each model is trained.
    :param train_inputs: Dictionary with the trainig input features for each network. The key is 'iX'
                         being X the network for which the data is and the value is the actual data.
    :param train_outputs: Dictionary with the trainig output labels for each network. The key is 'oX'
                          being X the network for which the data is and the value is the actual data.
    :param test_inputs: Dictionary with the testing input features for each network. The key is 'iX'
                        being X the network for which the data is and the value is the actual data.
    :param test_outputs: Dictionary with the testing output labels for each network. The key is 'oX'
                         being X the network for which the data is and the value is the actual data.
    :param complex: A boolean valeu that indicates if the network that is going to be evaluated is complex
                    or not. The conditions for being complex are: if is indicated in initialization, if 
                    user defined evalutaion method is used, if more than one network descriptor are 
                    evolved or if the network descriptor to be evolved is different from a MLP. It is complex
                    if occurs any of these conditions; otherwise, if none of the condition are true,
                    is is considered simple (not complex).
    :param toolbox: Obect Toolbox() from deap.base in the DEAP algorithm.
    :param selection: String that indicates the selection method used in the evolution. 
                      Possibilities are: 'best', 'tournament', 'roulette', 'random' or 'nsga2'.
    :param ev_alg: String that indicates the evolution algorithm that will be used.
                   Possibilities are: 'simple', 'mu_plus_lambda' or 'mu_comm_lambda'.
    :param evol_kwargs: Dictionary with parameters for the main DEAP function. The keys 
                        for that parameters are:' mu', 'lambda', 'cxpb' or 'mutpb'. Being
                        mu and lambda to float values for the mu_plus_lambda algorithm 
                        from DEAP. And cxpb and mutpb the crossover and mutation probabliites
                        respectively.
    :param cXp: Float value indicating the crossover probability. It will be 0 if there is only
                one descriptor to be evolved.
    :param mtp: Float value indicating the crossover probability. It will be 1 if there is only
                one descriptor to be evolved.
    :param generations: Number of generations that the evolution algorithm will be running.
    :param population_size: Population of individuals that will be evaluated in the evolution algorithm.
    :param ev_hypers: Hyperparameters to be evolved in the algorithm (e.g., optimizer, batch size).
    """

    def __init__(self, desc_list=[MLPDescriptor, ], compl=False, 
                 x_trains=None, y_trains=None, x_tests=None, y_tests=None, 
                 evaluation="XEntropy", n_inputs=((28, 28),), n_outputs=((10,),), 
                 batch_size=100, population=20, generations=20, iters=10, lrate=0.01, sel='best',
                 max_num_layers=10, max_num_neurons=100, max_filter=4, max_stride=3, seed=None , cxp=0, 
                 mtp=1, dropout=False, batch_norm=False, evol_kwargs={}, sel_kwargs={}, 
                 ev_alg='mu_plus_lambda', hyperparameters={}, custom_mutations={}, add_obj=0):

        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.max_num_layers = max_num_layers    
        self.max_num_neurons = max_num_neurons
        self.max_filter = max_filter
        self.max_stride = max_stride
        self.descriptors = desc_list
        self.evaluation = None                                          
        # Define and assign evalutaion function (both if is simple or complex case).
        self.define_evaluation(evaluation)

        self.batch_size = batch_size
        self.lrate = lrate
        self.iters = iters
        self.train_inputs = {}                                          
        self.train_outputs = {}
        self.test_inputs = {}
        self.test_outputs = {}
        # Load data in the previous dicts.
        self.data_save(x_trains, y_trains, x_tests, y_tests)            
        
        self.complex = self.is_complex(compl, evaluation, hyperparameters)
        
        self.toolbox = base.Toolbox()

        self.selection = None
        self.define_selection(sel)

        self.ev_alg = None
        self.evol_kwargs = evol_kwargs                                  
        self.cXp = cxp if len(desc_list) > 1 else 0
        self.mtp = mtp if len(desc_list) > 1 else 1

        self.generations = generations
        self.population_size = population
        self.ev_hypers = hyperparameters

        self.define_evolving(ev_alg)
        
        # Initialize DEAP-related matters.
        self.initialize_deap(sel, sel_kwargs, batch_norm, 
                             dropout, custom_mutations, add_obj)     
        
    def data_save(self, x_trains, y_trains, x_tests, y_tests):
        """
        Load data given by parameters in the atributes of data from
        the class Evolving, it initialices those atributes (train_inputs, 
        train_outputs, test_inputs and test_outputs).
        
        :param x_trains: Features data for training.
        :param y_trains: Labels data for training.
        :param x_tests: Features data for testing.
        :param y_tests: Labels data for testing.
        """
        if isinstance(x_trains, dict):
            self.train_inputs = x_trains
            self.train_outputs = y_trains

            self.test_inputs = x_tests
            self.test_outputs = y_tests
        else:
            for i, x in enumerate(x_trains):
                self.train_inputs["i" + str(i)] = x
                self.train_outputs["o" + str(i)] = y_trains[i]

            for i, x in enumerate(x_tests):
                self.test_inputs["i" + str(i)] = x
                self.test_outputs["o" + str(i)] = y_tests[i]

    def define_evaluation(self, evaluation):
        """
        Define the evaluation function. It accepts an string ('MSE' or 'XEntropy)
        and it will use the predifined functions from TensorFlow; this will be done
        in simple cases. Otherwise, the evaluation function will be defined and passed
        by parameter by the user.
        
        :param evaluation: Evaluation function. Either string (predefined) or customized by the user.
        """
        evals = {"MSE": tf.losses.mean_squared_error, "XEntropy": tf.nn.softmax_cross_entropy_with_logits}

        if type(evaluation) is str:
            self.evaluation = evals[evaluation]
        else:
            self.evaluation = evaluation
    
    def define_selection(self, selection):
        """
        Define the selection method for the evolution algorithm. It uses predefined methods
        from DEAP library.
        
        :param selection: String that indicates the selection method ('best', 'tournament', 'roulette',
                          'random' or 'nsga2').
        """
        sel_methods = {'best':tools.selBest, 'tournament':tools.selTournament, 'roulette':tools.selRoulette, 
                       'random':tools.selRandom, 'nsga2':tools.selNSGA2}
       
        if type(selection) is str:
            self.selection = sel_methods[selection]
        else:
            self.selection = selection
            
    def define_evolving(self, ev_alg):
        """
        Define the evolutionary algorithm for the evolution. It uses predefined algorithms
        from DEAP library.
        
        :param selection: String that indicates the evolutionary algorithm 
                          ('simple', 'mu_plus_lambda' or 'mu_comm_lambda').
        """
        deap_algs = {'simple':algorithms.eaSimple, 'mu_plus_lambda': algorithms.eaMuPlusLambda, 
                     'mu_comm_lambda':algorithms.eaMuCommaLambda}
        
        if type(ev_alg) is str:
            
            self.ev_alg = deap_algs[ev_alg]
        
            if not self.evol_kwargs:
                if ev_alg == 'simple':
                    self.evol_kwargs = {"cxpb": self.cXp, "mutpb": self.mtp}
                else:
                    self.evol_kwargs = {"mu": self.population_size, "lambda_": self.population_size, 
                                        "cxpb": self.cXp, "mutpb": self.mtp}
        else:
            self.ev_alg = ev_alg
    
    def is_complex(self, compl, evaluation, hyperparameters):
        """
        Determines if the case that will be evolved is a simple or a complex case.
        
        :param compl: A boolean value that directly indicates if is a simple or 
                      complex case.
        :param evaluation: Evaluation parameter that is used for initialization 
                           (string or function).
        :param hyperparameters: Dictionary with the hyperparameters to ve evolved.
        :return: True boolean value if is a complex case and False boolean value
                 if is a simple case.
        """
        if compl:
            return True
        elif type(evaluation) is not str:
            return True
        elif self.descriptors[0] is not MLPDescriptor:
            return True
        elif len(self.descriptors) > 1 or len(hyperparameters) > 0:
            return True
        
        return False
        
    
    def initialize_deap(self, sel, sel_kwargs, batch_norm, dropout, custom_mutations, add_obj):
        """
        Initialize DEAP function and atributes in order to be ready for evolutionary algorithm.
        
        :param sel: Selection method.
        :param sel_kwargs: Hyperparameters for the selection methods (e.g., size of the tournament 
                           if that method is selected).
        :param ev_alg: DEAP evolutionary algorithm.
        :param ev_kwargs: Hyperparameters for the evolutionary algorithm,(e.g., mutation or 
                          crossover probability).
        :param batch_norm: Whether the evolutive process includes batch normalization in the 
                           networks or not.
        :param dropout: Whether the evolutive process includes dropout in the networks or not.
        :param add_obj: Number of additional objectives.
        """

        creator.create("Fitness", base.Fitness, weights=[-1.0]*(len(self.test_outputs) + add_obj))

        creator.create("Individual", DescriptorContainer, fitness=creator.Fitness)

        self.toolbox.register("individual", self.init_individual, creator.Individual, batch_norm, dropout)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.eval_individual)
        self.toolbox.register("mate", cross, creator.Individual)
        self.toolbox.register("mutate", mutations, self.ev_hypers, self.max_num_neurons, custom_mutations)

        self.toolbox.register("select", self.selection, **sel_kwargs)


    def evolve(self):
        """
        Function that actualy applies the evolutionary algorithm. Using all the information
        provided in the initialization of the class, this function does the evolution. It will
        print the mean, standard, minimum and maximum values obtained form the individuals in
        each generation. Finally, it return the individuals from the lasta generation, the stats
        and the best individuals found during the algorithm.
        
        :return: The last generation, a log book (stats) and the hall of fame (the best 
                 individuals found).
        """

        pop = self.toolbox.population(n=self.population_size)
        hall_of = tools.HallOfFame(self.population_size)
        stats = tools.Statistics(lambda ind: ind.fitness.values)

        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        result, log_book = self.ev_alg(pop, self.toolbox, ngen=self.generations, 
                                       **self.evol_kwargs, verbose=1, 
                                       stats=stats, halloffame=hall_of)

        return result, log_book, hall_of

    def init_individual(self, init_ind, batch_norm, dropout):
        """
        Initializes the individual that is going to be used and evolved during the evolutionary
        algorithm. It will be used s dictionary with the string network id as key and the network 
        descriptor as a value, i.e., {"net_id": net_desc}.
        
        :param init_ind: DEAP function for transforming a network descriptor, or a list of 
                         descriptors + evolvable hyperparameters into a DEAP individual.
        :param batch_norm: A boolean value that indicates whether batch normalization is 
                           included into the evolution or not.
        :param dropout: A boolean value that incidates whether dropout is included into 
                        the evolution or not.
        :return: A DEAP individual totaly initialized.
        """

        network_descriptor = {}

        if not self.complex:  # Simple case
            network_descriptor["n0"] = MLPDescriptor()
            network_descriptor["n0"].random_init(self.train_inputs["i0"].shape[1:], self.train_outputs["o0"].shape[1], 
                                                 self.max_num_layers, self.max_num_neurons, None, None, dropout, batch_norm)
        else:  # Complex case
            for i, descriptor in enumerate(self.descriptors):
                network_descriptor["n" + str(i)] = descriptor()
                network_descriptor["n" + str(i)].random_init(self.n_inputs[i], self.n_outputs[i], self.max_num_layers, 
                                                             self.max_num_neurons, self.max_stride, self.max_filter, 
                                                             dropout, batch_norm)
        network_descriptor["hypers"] = {}
        if len(self.ev_hypers) > 0:

            for hyper in self.ev_hypers:
                network_descriptor["hypers"][hyper] = np.random.choice(self.ev_hypers[hyper])

        return init_ind(network_descriptor)

    def eval_individual(self, individual):
        """
        Function used for evaluating a DEAP individual during the evolutionary algorithm.
        
        :param individual: DEAP individual.
        :return: Value obtained from the evaluation.
        """
        if not self.complex:
            ev = self.single_net_eval(individual)
        else:
            ev = self.eval_multinetwork(individual)
        return ev

    def single_net_eval(self, individual):
        """
        Evaluation in the simple case. Function for evolving a single individual. 
        No need of the user providing a evaluation function.
        
        :param individual: DEAP individual
        :return: Value obtained from the evaluation.
        """
        net = MLP(individual.descriptor_list["n0"])
        
        inp = Input(shape=self.n_inputs[0])
        out = Flatten()(inp)
        out = net.building(out)
        model = Model(inputs=inp, outputs=out)

        opt = tf.keras.optimizers.Adam(learning_rate=self.lrate)
        model.compile(loss=self.evaluation, optimizer=opt, metrics=[])
        
        model.fit(self.train_inputs['i0'], self.train_outputs['o0'], epochs=self.iters, batch_size=self.batch_size, verbose=0)
        
        ev = model.evaluate(self.test_inputs['i0'], self.test_outputs['o0'], verbose=0)
        
        if isinstance(ev, float):
            ev = (ev,)
            
        return ev

    def eval_multinetwork(self, individual):
        """
        Evaluation in the complex case. Function for evolving individuals in a 
        complex case.  The user must have implemented the training and evaluation functions.
        
        :param individual: DEAP individual
        :return: Value obtained from the evaluation.
        """
        nets = {}
    
        for index, net in enumerate(individual.descriptor_list.keys()):
            if "hypers" not in net:
                nets[net] = descs[self.descriptors[index].__name__](individual.descriptor_list[net])

        ev = self.evaluation(nets, self.train_inputs, self.train_outputs, self.batch_size, self.iters,
                             self.test_inputs, self.test_outputs, individual.descriptor_list["hypers"])

        return ev


def mutations(ev_hypers, max_num_layers, custom_mutations, individual):
    """
    Mutation operators for individuals. They can affect any network or the hyperparameters.
    
    :param ev_hypers: Hyperparameters not included in the networks to be evolved.
    :param max_num_layers: Maximum number of layers in networks.
    :param batch_normalization: Whether batch normalization is part of the evolution or not.
    :param drop: Whether dropout is part of the evolution or not.
    :param individual: DEAP individual. Contains a dict where the keys are the components of the model.
    :return: Mutated version of the DEAP individual.
    """
    mutation_types = {'MLPDescriptor': MLP_Mutation, 'ConvDescriptor': CNN_Mutation, 
                      'TConvDescriptor': TCNN_Mutation, 'RNNDescriptor': RNN_Mutation}

    nets = list(individual.descriptor_list.keys())
    hyperparameters = individual.descriptor_list["hypers"]
    nets.remove("hypers")

    network = individual.descriptor_list[np.random.choice(nets)]

    if not custom_mutations:
        network_custom_mutations = [] # If no custom mutations are passed, each network's mutations will be applied
    else:
        network_custom_mutations = custom_mutations[network.__class__.__name__]
    
    network_mutation = mutation_types[network.__class__.__name__](ev_hypers, max_num_layers, network, 
                                                                  hyperparameters, network_custom_mutations)
    network_mutation.apply_random_mutation()
    
    return individual,

def cross(init_ind, ind1, ind2):
    """
    Crossover operator for individuals. Cannot be applied in the simple case, as it randomly 
    interchanges model components.
    
    :param init_ind: DEAP function for initializing dicts (in this case) as DEAP individuals.
    :param ind1: 1st individual to be crossed (first parent).
    :param ind2: 2st individual to be crossed (second parent).
    :return: Two new DEAP individuals, the crossed versions of the incoming 
             parameters (the offspring).
    """

    keys = list(ind1.descriptor_list.keys())
    # Randomly select the keys of the components that will be interchanged.
    cx_point = np.random.choice(keys, size=np.random.randint(1, len(keys)) if len(keys) > 2 else 1, replace=False)
    new1 = {}
    new2 = {}

    for key in keys:
        if key in cx_point:
            new1[key] = ind1.descriptor_list[key]
            new2[key] = ind2.descriptor_list[key]
        else:
            new1[key] = ind2.descriptor_list[key]
            new2[key] = ind1.descriptor_list[key]


    return init_ind(new1), init_ind(new2)
